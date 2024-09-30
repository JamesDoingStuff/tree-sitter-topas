(definition) @keyword
(comment) @comment
(string_literal) @string
(integer_literal) @number
(float_literal) @number

(macro_invocation name: (identifier) @function.macro)
(argument_list ( _ (identifier) @variable.parameter))

"@" @operator
"!" @operator
