// grd_writer.cpp — GRD write implementation
#include "grd_writer.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <zstd.h>

namespace grd {

// ── pwrite with ENOSPC/EIO retry ─────────────────────────────────────────────

static void safe_pwrite(int fd, const void* data, size_t len, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    off_t pos = static_cast<off_t>(offset);
    int retry_secs = 5;
    size_t written = 0;
    while (written < len) {
        ssize_t n = ::pwrite(fd, p + written, len - written,
                             pos + static_cast<off_t>(written));
        if (n < 0) {
            if (errno == EINTR) continue;
            if ((errno == ENOSPC || errno == EIO) && retry_secs <= 300) {
                spdlog::warn("grd: pwrite failed ({}), retrying in {}s",
                             strerror(errno), retry_secs);
                std::this_thread::sleep_for(std::chrono::seconds(retry_secs));
                retry_secs = std::min(retry_secs * 2, 300);
                continue;
            }
            throw std::runtime_error(std::string("grd: pwrite failed: ") +
                                     strerror(errno));
        }
        written += static_cast<size_t>(n);
    }
}

// ── GrdWriter ────────────────────────────────────────────────────────────────

GrdWriter::GrdWriter(const std::filesystem::path& path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0)
        throw std::runtime_error("grd: cannot open for writing: " + path.string());

    write_file_header();
}

GrdWriter::~GrdWriter() {
    if (!closed_ && fd_ >= 0) {
        try { close(); } catch (...) {}
    }
    if (fd_ >= 0) ::close(fd_);
}

void GrdWriter::write_file_header() {
    FileHeader hdr{};
    hdr.magic = GRD1_MAGIC;
    hdr.version_major = 1;
    hdr.version_minor = 0;
    auto now = std::chrono::system_clock::now();
    hdr.created_at_unix = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(now));
    append_bytes(&hdr, sizeof(hdr));
}

std::vector<uint8_t> GrdWriter::compress(const void* data, size_t size) const {
    size_t bound = ZSTD_compressBound(size);
    std::vector<uint8_t> out(bound);
    size_t csize = ZSTD_compress(out.data(), bound, data, size, 3);
    if (ZSTD_isError(csize))
        throw std::runtime_error(std::string("grd: zstd: ") +
                                 ZSTD_getErrorName(csize));
    out.resize(csize);
    return out;
}

uint64_t GrdWriter::write_section(uint32_t type, uint64_t section_id,
                                  const void* data, size_t size,
                                  uint64_t item_count,
                                  uint64_t aux0, uint64_t aux1) {
    auto compressed = compress(data, size);
    uint64_t sec_offset = offset_;
    append_bytes(compressed.data(), compressed.size());

    SectionDesc desc{};
    desc.type = type;
    desc.version = 1;
    desc.section_id = section_id;
    desc.file_offset = sec_offset;
    desc.compressed_size = compressed.size();
    desc.uncompressed_size = size;
    desc.item_count = item_count;
    desc.aux0 = aux0;
    desc.aux1 = aux1;
    sections_.push_back(desc);

    return sec_offset;
}

void GrdWriter::append_bytes(const void* data, size_t len) {
    safe_pwrite(fd_, data, len, offset_);
    offset_ += len;
}

void GrdWriter::pwrite_bytes(const void* data, size_t len, uint64_t offset) {
    safe_pwrite(fd_, data, len, offset);
}

uint32_t GrdWriter::intern_string(const std::string& s) {
    auto it = string_index_.find(s);
    if (it != string_index_.end()) return it->second;
    uint32_t idx = static_cast<uint32_t>(strings_.size());
    strings_.push_back(s);
    string_index_[s] = idx;
    return idx;
}

// ── PCA 3D projection ────────────────────────────────────────────────────────
// Computes the first 3 principal components of the embedding matrix.
// Inputs are L2-normalized (on unit sphere), so we center and extract
// the top-3 eigenvectors of the covariance matrix.

std::vector<float> GrdWriter::compute_pca3(const float* embeddings,
                                           uint32_t n, uint32_t dim) {
    using namespace Eigen;

    // Map row-major float data into Eigen matrix
    MatrixXf X(n, dim);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < dim; ++j)
            X(i, j) = embeddings[i * dim + j];

    // Center
    RowVectorXf mean = X.colwise().mean();
    X.rowwise() -= mean;

    // For large n, use randomized SVD via the Gram matrix X*X^T (n×n)
    // For small n or when dim < n, use covariance X^T*X (dim×dim)
    std::vector<float> result(n * 3);

    if (n <= dim) {
        // Gram matrix path: n×n
        MatrixXf G = X * X.transpose();
        SelfAdjointEigenSolver<MatrixXf> solver(G);
        // Eigenvalues are sorted ascending; take last 3
        int ncols = solver.eigenvectors().cols();
        int start = std::max(0, ncols - 3);
        MatrixXf proj(n, 3);
        for (int c = 0; c < 3; ++c) {
            int eigcol = ncols - 1 - c; // descending order
            if (eigcol >= 0)
                proj.col(c) = solver.eigenvectors().col(eigcol);
            else
                proj.col(c).setZero();
        }
        // Normalize each point onto unit sphere for visualization
        for (uint32_t i = 0; i < n; ++i) {
            float norm = proj.row(i).norm();
            if (norm > 1e-8f) proj.row(i) /= norm;
            result[i * 3 + 0] = proj(i, 0);
            result[i * 3 + 1] = proj(i, 1);
            result[i * 3 + 2] = proj(i, 2);
        }
    } else {
        // Covariance matrix path: dim×dim
        MatrixXf cov = (X.transpose() * X) / static_cast<float>(n - 1);
        SelfAdjointEigenSolver<MatrixXf> solver(cov);
        int ncols = solver.eigenvectors().cols();
        // Project onto top-3 eigenvectors
        MatrixXf V(dim, 3);
        for (int c = 0; c < 3; ++c) {
            int eigcol = ncols - 1 - c;
            if (eigcol >= 0)
                V.col(c) = solver.eigenvectors().col(eigcol);
            else
                V.col(c).setZero();
        }
        MatrixXf proj = X * V; // n×3
        // Normalize onto unit sphere
        for (uint32_t i = 0; i < n; ++i) {
            float norm = proj.row(i).norm();
            if (norm > 1e-8f) proj.row(i) /= norm;
            result[i * 3 + 0] = proj(i, 0);
            result[i * 3 + 1] = proj(i, 1);
            result[i * 3 + 2] = proj(i, 2);
        }
    }

    return result;
}

// ── write_taxon ──────────────────────────────────────────────────────────────

void GrdWriter::write_taxon(const TaxonData& data) {
    std::lock_guard lock(mu_);

    const uint32_t n = static_cast<uint32_t>(data.accessions.size());
    const uint32_t taxon_ord = next_id_++;
    const uint64_t sid_base = static_cast<uint64_t>(taxon_ord) * 16;

    // Count reps and contaminated
    uint32_t n_reps = 0, n_contam = 0;
    for (auto s : data.status) {
        if (s == GenomeStatus::REPRESENTATIVE) ++n_reps;
        if (s == GenomeStatus::CONTAMINATED) ++n_contam;
    }

    // ── TMTA: taxon metadata ─────────────────────────────────────────────────
    {
        TaxonMetaHeader hdr{};
        hdr.magic = SEC_TMTA;
        hdr.n_genomes = n;
        hdr.n_reps = n_reps;
        hdr.n_contaminated = n_contam;
        hdr.embed_dim = data.embed_dim;
        hdr.sketch_size = data.sketch_size;
        hdr.kmer_size = data.kmer_size;
        hdr.k_conn = data.k_conn;
        hdr.diversity_threshold = data.diversity_threshold;
        hdr.ani_threshold = data.ani_threshold;
        hdr.mst_p90_edge = data.mst_p90_edge;
        hdr.mst_true_max = data.mst_true_max;

        // Payload: header + taxonomy string (length-prefixed)
        std::vector<uint8_t> payload(sizeof(TaxonMetaHeader));
        std::memcpy(payload.data(), &hdr, sizeof(hdr));
        uint32_t tax_len = static_cast<uint32_t>(data.taxonomy.size());
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(&tax_len),
                       reinterpret_cast<const uint8_t*>(&tax_len) + 4);
        payload.insert(payload.end(),
                       data.taxonomy.begin(), data.taxonomy.end());

        write_section(SEC_TMTA, sid_base + 0,
                      payload.data(), payload.size(), n,
                      data.embed_dim, data.kmer_size);
    }

    // ── GMET: per-genome metadata (columnar) ─────────────────────────────────
    {
        std::vector<uint8_t> payload;
        // Accession string table: offsets + data
        std::vector<uint32_t> acc_offsets(n);
        std::string acc_data;
        for (uint32_t i = 0; i < n; ++i) {
            acc_offsets[i] = static_cast<uint32_t>(acc_data.size());
            acc_data += data.accessions[i];
            acc_data.push_back('\0');
        }
        // n_genomes
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(&n),
                       reinterpret_cast<const uint8_t*>(&n) + 4);
        // accession offsets
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(acc_offsets.data()),
                       reinterpret_cast<const uint8_t*>(acc_offsets.data() + n));
        // accession data
        payload.insert(payload.end(), acc_data.begin(), acc_data.end());
        // status
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.status.data()),
                       reinterpret_cast<const uint8_t*>(data.status.data() + n));
        // nearest_rep_idx
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.nearest_rep_idx.data()),
                       reinterpret_cast<const uint8_t*>(data.nearest_rep_idx.data() + n));
        // nearest_rep_dist
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.nearest_rep_dist.data()),
                       reinterpret_cast<const uint8_t*>(data.nearest_rep_dist.data() + n));
        // component_id
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.component_id.data()),
                       reinterpret_cast<const uint8_t*>(data.component_id.data() + n));
        // outlier_zscore
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.outlier_zscore.data()),
                       reinterpret_cast<const uint8_t*>(data.outlier_zscore.data() + n));
        // genome_length
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(data.genome_length.data()),
                       reinterpret_cast<const uint8_t*>(data.genome_length.data() + n));

        write_section(SEC_GMET, sid_base + 1,
                      payload.data(), payload.size(), n);
    }

    // ── EMBD: full embeddings ────────────────────────────────────────────────
    if (!data.embeddings.empty()) {
        write_section(SEC_EMBD, sid_base + 2,
                      data.embeddings.data(),
                      data.embeddings.size() * sizeof(float),
                      n, data.embed_dim);
    }

    // ── PRJ3: 3D PCA projection ──────────────────────────────────────────────
    if (!data.embeddings.empty() && n >= 3) {
        auto proj = compute_pca3(data.embeddings.data(), n, data.embed_dim);
        write_section(SEC_PRJ3, sid_base + 3,
                      proj.data(), proj.size() * sizeof(float), n);

        // Compute centroid for meta-sphere
        float cx = 0, cy = 0, cz = 0;
        for (uint32_t i = 0; i < n; ++i) {
            cx += proj[i * 3 + 0];
            cy += proj[i * 3 + 1];
            cz += proj[i * 3 + 2];
        }
        float inv_n = 1.0f / static_cast<float>(n);
        TaxonIndexEntry tidx{};
        tidx.taxonomy_hash = fnv1a_64(data.taxonomy);
        tidx.strtable_offset = intern_string(data.taxonomy);
        tidx.n_genomes = n;
        tidx.n_reps = n_reps;
        tidx.n_contaminated = n_contam;
        tidx.section_id_base = sid_base;
        tidx.centroid_3d[0] = cx * inv_n;
        tidx.centroid_3d[1] = cy * inv_n;
        tidx.centroid_3d[2] = cz * inv_n;
        taxon_index_.push_back(tidx);
    } else {
        // No embeddings — still register in taxon index
        TaxonIndexEntry tidx{};
        tidx.taxonomy_hash = fnv1a_64(data.taxonomy);
        tidx.strtable_offset = intern_string(data.taxonomy);
        tidx.n_genomes = n;
        tidx.n_reps = n_reps;
        tidx.n_contaminated = n_contam;
        tidx.section_id_base = sid_base;
        taxon_index_.push_back(tidx);
    }

    // ── EDGE: rep→member edges ───────────────────────────────────────────────
    if (!data.edges.empty()) {
        write_section(SEC_EDGE, sid_base + 4,
                      data.edges.data(),
                      data.edges.size() * sizeof(EdgeEntry),
                      data.edges.size());
    }

    // ── Build cross-taxon accession index entries ────────────────────────────
    for (uint32_t i = 0; i < n; ++i) {
        AccessionIndexEntry ae{};
        ae.accession_hash = fnv1a_64(data.accessions[i]);
        ae.taxon_ordinal = taxon_ord;
        ae.genome_idx = i;
        accession_index_.push_back(ae);
    }

    total_genomes_ += n;
    taxonomy_strings_.push_back(data.taxonomy);
}

// ── close: write global sections + TOC + TailLocator ─────────────────────────

void GrdWriter::close() {
    if (closed_) return;
    closed_ = true;

    // ── TIDX: taxon directory (sorted by taxonomy_hash) ──────────────────────
    std::sort(taxon_index_.begin(), taxon_index_.end(),
              [](const TaxonIndexEntry& a, const TaxonIndexEntry& b) {
                  return a.taxonomy_hash < b.taxonomy_hash;
              });
    write_section(SEC_TIDX, 0xFFFF'0000,
                  taxon_index_.data(),
                  taxon_index_.size() * sizeof(TaxonIndexEntry),
                  taxon_index_.size());

    // ── ACCX: cross-taxon accession index (sorted by hash) ───────────────────
    std::sort(accession_index_.begin(), accession_index_.end(),
              [](const AccessionIndexEntry& a, const AccessionIndexEntry& b) {
                  return a.accession_hash < b.accession_hash;
              });
    write_section(SEC_ACCX, 0xFFFF'0001,
                  accession_index_.data(),
                  accession_index_.size() * sizeof(AccessionIndexEntry),
                  accession_index_.size());

    // ── STRT: string table ───────────────────────────────────────────────────
    {
        std::vector<uint8_t> raw;
        uint32_t n_strings = static_cast<uint32_t>(strings_.size());
        raw.insert(raw.end(),
                   reinterpret_cast<const uint8_t*>(&n_strings),
                   reinterpret_cast<const uint8_t*>(&n_strings) + 4);
        for (const auto& s : strings_) {
            raw.insert(raw.end(), s.begin(), s.end());
            raw.push_back('\0');
        }
        write_section(SEC_STRT, 0xFFFF'0002,
                      raw.data(), raw.size(), n_strings);
    }

    // ── TREE: taxonomy hierarchy ─────────────────────────────────────────────
    // Build a hierarchical tree from all taxonomy strings.
    // Format: "d__X;p__Y;c__Z;..." — split on ';' for each rank level.
    {
        struct TempNode {
            std::string name;
            uint32_t parent = UINT32_MAX;
            std::vector<uint32_t> children;
            uint32_t n_genomes = 0;
            uint32_t n_species = 0;
            uint8_t rank = 0;
        };
        std::vector<TempNode> nodes;
        std::unordered_map<std::string, uint32_t> path_to_node;

        // Root node
        TempNode root;
        root.name = "root";
        nodes.push_back(root);
        path_to_node[""] = 0;

        for (const auto& tidx : taxon_index_) {
            // Find taxonomy string from strtable_offset
            if (tidx.strtable_offset >= strings_.size()) continue;
            const std::string& taxonomy = strings_[tidx.strtable_offset];

            // Split taxonomy by ';'
            std::vector<std::string> parts;
            size_t pos = 0;
            while (pos < taxonomy.size()) {
                size_t next = taxonomy.find(';', pos);
                if (next == std::string::npos) next = taxonomy.size();
                parts.push_back(taxonomy.substr(pos, next - pos));
                pos = next + 1;
            }

            // Walk/create path
            std::string path;
            uint32_t parent_idx = 0;
            for (size_t r = 0; r < parts.size(); ++r) {
                if (!path.empty()) path += ";";
                path += parts[r];

                auto it = path_to_node.find(path);
                if (it == path_to_node.end()) {
                    uint32_t idx = static_cast<uint32_t>(nodes.size());
                    TempNode node;
                    node.name = parts[r];
                    node.parent = parent_idx;
                    node.rank = static_cast<uint8_t>(r);
                    nodes.push_back(node);
                    nodes[parent_idx].children.push_back(idx);
                    path_to_node[path] = idx;
                    parent_idx = idx;
                } else {
                    parent_idx = it->second;
                }
            }
            // Leaf node gets genome count
            nodes[parent_idx].n_genomes += tidx.n_genomes;
            nodes[parent_idx].n_species += 1;
        }

        // Propagate genome/species counts up
        // Process in reverse order (leaves first since higher indices = deeper)
        for (int i = static_cast<int>(nodes.size()) - 1; i > 0; --i) {
            auto& node = nodes[i];
            if (node.parent != UINT32_MAX) {
                nodes[node.parent].n_genomes += node.n_genomes;
                nodes[node.parent].n_species += node.n_species;
            }
        }

        // Serialize into TreeNode array
        std::vector<TreeNode> tree_nodes(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto& tn = tree_nodes[i];
            tn.strtable_offset = intern_string(nodes[i].name);
            tn.parent_idx = nodes[i].parent;
            tn.n_children = static_cast<uint32_t>(nodes[i].children.size());
            tn.first_child_idx = nodes[i].children.empty()
                                     ? UINT32_MAX
                                     : nodes[i].children[0];
            tn.n_genomes_subtree = nodes[i].n_genomes;
            tn.n_species_subtree = nodes[i].n_species;
            tn.rank = nodes[i].rank;
        }

        // Write child index arrays after tree nodes for navigation
        // Layout: [TreeNode[n_nodes]] [uint32_t children_of_0...] [children_of_1...] ...
        std::vector<uint8_t> tree_payload;
        uint32_t n_nodes = static_cast<uint32_t>(nodes.size());
        tree_payload.insert(tree_payload.end(),
                            reinterpret_cast<const uint8_t*>(&n_nodes),
                            reinterpret_cast<const uint8_t*>(&n_nodes) + 4);
        tree_payload.insert(tree_payload.end(),
                            reinterpret_cast<const uint8_t*>(tree_nodes.data()),
                            reinterpret_cast<const uint8_t*>(tree_nodes.data() + n_nodes));
        // Append child index arrays
        for (const auto& node : nodes) {
            if (!node.children.empty()) {
                tree_payload.insert(tree_payload.end(),
                                    reinterpret_cast<const uint8_t*>(node.children.data()),
                                    reinterpret_cast<const uint8_t*>(
                                        node.children.data() + node.children.size()));
            }
        }

        write_section(SEC_TREE, 0xFFFF'0003,
                      tree_payload.data(), tree_payload.size(), n_nodes);
    }

    // ── TOC ──────────────────────────────────────────────────────────────────
    uint64_t toc_offset = offset_;
    {
        TocHeader toc{};
        toc.magic = TOCB_MAGIC;
        toc.version = 1;
        toc.section_count = sections_.size();
        toc.n_taxa = taxon_index_.size();
        toc.n_genomes_total = total_genomes_;

        std::vector<uint8_t> toc_raw;
        toc_raw.insert(toc_raw.end(),
                       reinterpret_cast<const uint8_t*>(&toc),
                       reinterpret_cast<const uint8_t*>(&toc) + sizeof(toc));
        toc_raw.insert(toc_raw.end(),
                       reinterpret_cast<const uint8_t*>(sections_.data()),
                       reinterpret_cast<const uint8_t*>(
                           sections_.data() + sections_.size()));

        auto compressed = compress(toc_raw.data(), toc_raw.size());
        append_bytes(compressed.data(), compressed.size());
    }
    uint64_t toc_size = offset_ - toc_offset;

    // ── TailLocator ──────────────────────────────────────────────────────────
    {
        TailLocator tail{};
        tail.magic = GRDT_MAGIC;
        tail.version = 1;
        tail.toc_offset = toc_offset;
        tail.toc_size = toc_size;
        tail.n_taxa = taxon_index_.size();
        append_bytes(&tail, sizeof(tail));
    }

    ::fsync(fd_);
    spdlog::info("grd: wrote {} taxa, {} genomes, {} sections, {} bytes",
                 taxon_index_.size(), total_genomes_,
                 sections_.size(), offset_);
}

} // namespace grd
