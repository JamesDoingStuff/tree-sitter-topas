(definition) @keyword
(comment) @comment
(string_literal) @string
(integer_literal) @number
(float_literal) @number

(macro_invocation name: (identifier) @function.macro)

[
    "@"
    "!" 
] @operator

( _ directive: _ @keyword.preprocessor)

(binary_expression operator: _ @operator)

(unary_expression "-" @operator)

(identifier) @variable.parameter
