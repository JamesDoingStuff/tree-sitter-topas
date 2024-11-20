(definition) @keyword
(line_comment) @comment
(block_comment) @comment
(arrow_comment) @comment.todo

(string_literal) @string
(integer_literal) @number
(float_literal) @number

(macro_invocation name: (identifier) @function.macro)

[
    "@"
    "!" 
    "&"
] @operator

"macro" @keyword.preprocessor
(macro_declaration name: (identifier) @function.macro)
(macro_list name: (identifier) @function.macro)

( _ directive: _ @keyword.preprocessor)

(binary_expression operator: _ @operator)

(unary_expression "-" @operator)

(identifier) @variable.parameter
