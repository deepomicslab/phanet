# Resource files and packaging

The manual `20261008-v1` resource bundle contains 22 files and approximately
26.12 GB of payload. NumPy `.npy` files store the network arrays.

| Files | Purpose | Needed for |
|---|---|---|
| `glm/model.json`, `glm/weights.pt` | GLM configuration and inference weights (0.323 GB) | GLM / combined |
| `network/graph_data.npy`, `graph_indices.npy`, `graph_indptr.npy` | Edge weights, endpoint indices and row offsets of the fixed graph (15.028 GB) | Network / combined |
| `network/vectors.npy` | Normalized reference structure representations (5.834 GB) | Network / combined |
| `network/references.faa`, `self_bits.npy` | Reference sequences and self-alignment scores for sequence searches (1.180 GB) | Network / combined |
| `network/ids.npy`, `labels.npy` | Reference identities and functional labels (0.188 GB) | Network / combined |
| `network/support.npy` | Precomputed propagation masses (0.228 GB) | Network / combined |
| `network/profile_indices.npy`, `profile_indptr.npy`, `profile_coverage.npy`, `families.json` | Family membership, coverage and names (0.051 GB) | Network / combined |
| `network/network.json` | Version, dimensions, resource identities and checksums | Network / combined |
| `profiles/phrog_v4.hmm` | Third-party PHROG alignments converted to the required HMMER format (3.291 GB) | Current network / combined installer |
| `RESOURCE_MANIFEST.json` | Inventory and checksums used by the installer | Setup |
| `SHA256SUMS` | Convenient standalone integrity check | Distribution verification |
| `README.md`, `LICENSE`, `RESOURCE_NOTICES.md` | Instructions, license and upstream attribution | Distribution documentation |

GLM-only setup uses the GLM files. Network and combined modes use all network arrays and
the PHROG library.

## PHROG profiles

Use `profiles/phrog_v4.hmm` from the shared resource bundle. This converted third-party
file is tied to the reference network by its SHA-256. PhaNet does not download or
convert PHROG during installation. The publisher's HH-suite `.hhm` archive cannot
replace this HMMER library. Attribution and source terms are described in [resource
sources](resource-sources.md).
