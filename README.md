# matrix

Simple & extensible type-checked matrixes

This was made as a learning project and thrives to provide matrices generic over any type.
Some basic matrix manipulation operations are implemented for the matrix assuming the concrete type implements the required traits.
The main selling point is that most operations fail to compile if the operation is impossible. This is done through the use of `min_const_generic` (rustc v1.51) and allows operations such as matrix product to always work if the code compiles.

Although most operations implemented are done so with mathematical matrices in mind the type itself can be used for any use-case

For how to use this crate, refer to [`Matrix`]
