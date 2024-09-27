/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name:'topas',

  rules: {
    source_file: $ => repeat(/\s/),
}
});

