# Vendored dependencies

## robin_hood.h

- Project: `martinus/robin-hood-hashing`
- Source: `src/include/robin_hood.h`
- Commit: `7697343363af4cc3f42cab17be49e6af9ab181e2`
- Version declared by the header: `3.11.5`
- SHA-256: `ac6260e8070d1252fc993f3c95aefd0a52674cb5a1bad67539ead15b4614696b`
- License: MIT (the complete notice is retained in `robin_hood.h`)

The commit is pinned here so experiment builds do not depend on the mutable
`master` URL used to import the file.

## concurrentqueue.hpp

- Project: `cameron314/concurrentqueue`
- Source: `concurrentqueue.h`
- Imported unchanged from the local PrediCache dependency's `third_party/concurrentqueue.hpp`.
- SHA-256: `524bc2de581ecc95e632ee6aac2676d6a1029a692a9c1b4a32e4a80608abad5e`
- License: Simplified BSD (the complete notice is retained in the header).

Calico vendors this header directly so its core build does not require a separate
PrediCache checkout.
