## Tensor Functions

The tensor functions namespace holds the following functions:

| Function     | Description               |
| ------------ | ------------------------- |
| Scale        | y = αy                    |
| Update       | y = y + αx                |
| Fixed Update | y = y + α                 |
| Sum          | Sum two arrays            |
| Mean         | Mean of two arrays        |
| Min          | Minimum value in an array |
| Max          | Maximum value in an array |


Fixed update and sum are very similar operations, but rather than allocating an entire array of values, fixed update uses only one.
