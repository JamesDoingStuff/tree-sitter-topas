/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PRECEDENCE = {
  exponentiation: 5,
  unary: 4,
  multiplicative: 3,
  additive: 2,
  comparative: 1,
};

const atoms = require('./grammar/atoms.js');

const keywords = require('./grammar/keywords.js');

module.exports = grammar({
  name: 'topas',

  extras: $ => [
    /\s|\n/,
    $.line_comment,
    $.block_comment,
  ],

  rules: {
    source_file: $ => repeat(choice($.macro_invocation, $._Ttop, $.definition, $._literal, $._global_preprocessor_directive)),

    line_comment: $ => /'.*/,

    block_comment: $ => seq(
      '/*',
      repeat(/./),
      '*/',
    ),

    ...keywords,

    _literal: $ => choice($.string_literal, $.integer_literal, $.float_literal),

    string_literal: $ => /".*"/, // Anything between quote marks e.g., "a word"

    site_name_string: $ => /\S+/,

    site_query_string: $ => token(choice(
      seq(optional(/\S+/), '*'),
      seq('"', repeat(seq(optional('!'), /\S+/, optional('*'))), '"'),
    )),

    ...atoms,

    integer_literal: $ => /-?\d+/,

    float_literal: $ => choice(
      /-?\d*\.\d+/, // Ordinary floats e.g., 1.23
      /-?\d+(\.\d+)?(e|E)-?\d+(\.\d+)?/, // Scientific notation e.g., 1.3e4 or 2e5.5
    ),

    macro_invocation: $ => seq(
      field('name', $.identifier),
      field('arguments', prec(1, optional($.argument_list))),
    ),

    identifier: $ => /[A-Za-z]\w*/, // Initial letter character, then any alpha-numeric or underscore characters are permitted

    _argument: $ => choice(
      $.refined_parameter,
      $.unrefined_parameter,
      $._expression,
    ),

    argument_list: $ => seq(
      token.immediate('('),
      optional($._argument),
      repeat(seq(',', optional($._argument))), // N.B. Empty arguments *are* permitted
      ')',
    ),

    refined_parameter: $ => prec.left(seq(
      '@',
      optional($.identifier), // Parameter name, ignored internally
      optional(choice($.integer_literal, $.float_literal)), // Initial value
    )),

    unrefined_parameter: $ => prec.left(seq('!', optional(choice($.identifier, $._literal)))),

    _refineable_value_expression: $ => choice(
      $.simple_assignment,
      refineable(seq(optional($.identifier), $._literal)),
    ),

    _fixed_value_expression: $ => choice(
      $.simple_assignment,
      seq(optional($.identifier), $._literal),
    ),

    simple_assignment: $ => equation('=', $),

    compound_assignment: $ => equation(choice('+=', '-=', '*=', '/=', '^='), $),

    // ------ Expressions ------- //

    _expression: $ => choice(
      $._closed_expression,
      $.unary_expression,
      $.binary_expression,
    ),

    _closed_expression: $ => choice(
      prec(1, $.identifier),
      $.macro_invocation,
      $.parenthesised_expression,
      $._literal,
    ),

    _exponentiation: $ => prec.left(PRECEDENCE.exponentiation, seq(
      field('left', choice(
        $._closed_expression,
        alias($._exponentiation, $.binary_expression),
      )),
      field('operator', '^'),
      field('right', $._expression),
    )),

    binary_expression: $ => choice(

      $._exponentiation,

      prec.left(PRECEDENCE.multiplicative, seq(
        field('left', $._expression),
        field('operator', choice('*', '/', '%')),
        field('right', $._expression),
      )),

      prec.left(PRECEDENCE.multiplicative, seq(
        field('left', $._expression),
        field('right', choice(
          $._closed_expression,
          alias($._exponentiation, $.binary_expression),
        )),
      )),

      prec.left(PRECEDENCE.additive, seq(
        field('left', $._expression),
        field('operator', choice('+', '-')),
        field('right', $._expression),
      )),

      prec.left(PRECEDENCE.comparative, seq(
        field('left', $._expression),
        field('operator', choice('==', '<', '>', '<=', '>=')),
        field('right', $._expression),
      )),
    ),

    unary_expression: $ => prec.right(PRECEDENCE.unary, seq(
      field('operator', choice('+', '-')),
      field('argument', choice(
        $.identifier,
        $.parenthesised_expression,
        $.float_literal,
        $.integer_literal,
        alias($._exponentiation, $.binary_expression),
      )),
    )),

    parenthesised_expression: $ => seq(
      '(',
      $._expression,
      ')',
    ),

    _block_item: $ => prec.right(choice(
      $.definition,
      $._global_preprocessor_directive,
      $._Ttop,
      $._expression,
      $.refined_parameter,
      $.unrefined_parameter,
      $.simple_assignment,
      $.compound_assignment,
    )),

    // ------- Preprocessor -------- //

    parameter_list: $ => seq(
      token.immediate('('),
      optional('&'),
      optional($.identifier),
      repeat(seq(',', optional('&'), optional($.identifier))),
      ')',
    ),

    macro_declaration: $ => seq(
      token('macro'),
      optional('&'),
      field('name', $.identifier),
      field('parameters', optional($.parameter_list)),
      field('body', seq(
        '{',
        repeat(choice($._block_item, $._macro_preprocessor_directive)),
        '}',
      )),
    ),

    _global_preprocessor_directive: $ => choice(
      $.macro_declaration,
      $.preprocessor_include,
      $.preprocessor_delete,
      $.preprocessor_define,
      $.preprocessor_call,
      $.preprocessor_if_statement,
      $.preprocessor_variable_declaration,
      $.preprocessor_output,
      $.macro_list,
    ),

    preprocessor_include: $ => seq(field('directive', '#include'), field('path', $.string_literal)),
    preprocessor_delete: $ => seq(field('directive', '#delete_macros'), field('arguments', seq('{', repeat($.identifier), '}'))),
    preprocessor_define: $ => seq(field('directive', choice('#define', '#undef')), field('argument', $.identifier)),
    preprocessor_call: $ => field('directive', '#seed'),

    preprocessor_if_statement: $ => seq(
      choice(
        $._preproc_if,
        $._preproc_ifdef,
        $._preproc_ifndef,
      ),
      repeat($._preproc_elseif),
      optional($._preproc_else),
      field('directive', '#endif'),
    ),

    _preproc_if: $ => seq(
      field('directive', '#if'),
      optional('='),
      field('condition', $._expression),
      ';',
      optional(repeat($._block_item)),
    ),

    _preproc_ifdef: $ => seq(
      field('directive', '#ifdef'),
      optional('!'),
      field('argument', $.identifier),
      optional(repeat($._block_item)),
    ),

    _preproc_ifndef: $ => seq(
      field('directive', '#ifndef'),
      field('argument', $.identifier),
      optional(repeat($._block_item)),
    ),

    _preproc_elseif: $ => seq(
      field('directive', '#elseif'),
      optional('='),
      field('condition', $._expression),
      ';',
      optional(repeat($._block_item)),
    ),

    _preproc_else: $ => seq(
      field('directive', '#else'),
      optional(repeat($._block_item)),
    ),

    macro_list: $ => seq(
      field('directive', '#list'),
      repeat1(seq(
        optional('&'),
        field('name', $.identifier),
        field('parameters', optional($.parameter_list)),
      )),
      field('body', seq(
        '{',
        repeat(choice(
          $._literal,
          $.identifier,
          $.refined_parameter,
          $.unrefined_parameter,
          $.delimited_block,
        )),
        '}',
      )),
    ),

    delimited_block: $ => seq(
      '{',
      repeat(choice(
        $._block_item,
        $._macro_preprocessor_directive,
      )),
      '}',
    ),

    preprocessor_variable_declaration: $ => seq(
      field('directive', '#prm'),
      field('name', $.identifier),
      field('value', $.simple_assignment),
    ),

    preprocessor_output: $ => seq(
      field('directive', '#out'),
      field('argument', $.identifier),
    ),

    _macro_preprocessor_directive: $ => choice(
      $.macro_if_statement,
      $.macro_operator_directive,
      alias($._macro_unique, $.macro_operator_directive),
      $.macro_parameter_output,
    ),

    macro_if_statement: $ => seq(
      choice(
        $._m_if,
        $._m_ifarg,
      ),
      repeat($._m_elseif),
      optional($._m_else),
      field('directive', '#m_endif'),
    ),

    _m_if: $ => seq(
      field('directive', '#m_if'),
      field('argument', $.binary_expression),
      ';',
      optional(repeat($._block_item)),
    ),

    _m_elseif: $ => seq(
      field('directive', '#m_elseif'),
      field('argument', $.binary_expression),
      ';',
      optional(repeat($._block_item)),
    ),

    _m_ifarg: $ => seq(
      field('directive', '#m_ifarg'),
      field('argument', $.identifier),
      choice(
        field('directive', choice('#m_code', '#_eqn', '#m_code_refine', '#m_one_word')),
        $.string_literal,
      ),
      optional(repeat($._block_item)),
    ),

    _m_else: $ => seq(
      field('directive', '#m_else'),
      optional(repeat($._block_item)),
    ),

    macro_operator_directive: $ => choice(
      field('directive', choice('#m_argu', '#m_first_word', '#m_unique_not_refine')),
      field('argument', $.identifier),
    ),

    _macro_unique: $ => field('directive', '#m_unique'),

    macro_parameter_output: $ => seq(
      field('directive', '#m_out'),
      field('argument', $.identifier),
    ),

    // ------- Keywords -------- //

    _Ttop: $ => choice(
      $._Tcomm_2,
      $._Tstr_details, // At top-level temporarily for test purposes, until [str] keyword is added
    ),

    _Tcomm_2: $ => choice(
      $.variable_declaration,
      $.variable_assignment,
    ),

    _Tstr_details: $ => choice(
      $.site_declaration,
    ),

    _Tmin_r_max_r: $ => choice(
      simple_keyword('min_r', $, false),
      simple_keyword('max_r', $, false),
    ),

    variable_declaration: $ => seq(
      field('keyword', choice('prm', 'local')),
      field('name', refineable($.identifier)),
      field('value', choice($._literal, $.simple_assignment)),
    ),

    variable_assignment: $ => seq(
      field('keyword', 'existing_prm'),
      field('name', refineable($.identifier)),
      field('value', choice(
        $.simple_assignment,
        $.compound_assignment,
      )),
    ),

    /*

    [site [x][y][z]]...
      [occ [beq][scale_occ]]...
      [num_posns][rand_xyz][inter]
      [adps][u11][u22][u33][u12][u13][u23]

    */

    site_declaration: $ => prec.right(seq(
      field('keyword', 'site'),
      field('name', $.site_name_string),
      repeat(choice(
        alias($._site_nested_keyword, $.keyword_statement),
        alias($._Tmin_r_max_r, $.keyword_statement),
        $._global_preprocessor_directive,
        prec(-1, $.macro_invocation),
      )),
    )),

    _site_nested_keyword: $ => choice(
      simple_keyword(/[xyz]/, $),
      $._occ_keyword_statement,
      simple_keyword('num_posns', $),
      simple_keyword('rand_xyz', $, false),
      simple_keyword('inter', $, false),
      choice(
        field('keyword', 'adps'),
        prec.right(repeat1(simple_keyword(/u(1[123]|2[23]|33)/, $))),
      ),
    ),

    _occ_keyword_statement: $ => seq(
      field('keyword', 'occ'),
      $.atom,
      $._refineable_value_expression,
      repeat(alias($._occ_nested_keyword, $.keyword_statement)),
    ),

    _occ_nested_keyword: $ => choice(
      simple_keyword('beq', $),
      simple_keyword('scale_occ', $),
    ),
  },
});


/**
 * Creates a rule to optionally allow or disallow refinement of a rule
 *
 * @param {Rule} rule
 *
 * @returns {ChoiceRule}
 */
function refineable(rule) {
  return choice(
    seq(optional(choice('@', '!')), rule),
    prec(-1, '@'),
  );
}

/**
 * Creates a rule for the TOPAS equation structure
 *
 * @param {Rule | string} operator
 *
 * @param {GrammarSymbols<string>} $
 *
 * @returns {SeqRule}
 */
function equation(operator, $) {
  return seq(
    field('operator', operator),
    field('body', $._expression),
    ';',
    optional(seq(':', choice($.float_literal, $.integer_literal, $.identifier))),
  );
}

/**
 * Creates a rule for the structure of simple keywords i.e. those that take
 * only an equation or value and have no nested keywords beneath them. Refinement
 * symbols are allowed by default but can be disallowed using third argument.
 *
 * @param {string | RegExp} keyword
 * @param {GrammarSymbols<string>} $
 * @param {boolean} refine
 * @returns {SeqRule}
 */
function simple_keyword(keyword, $, refine=true) {
  if (refine) {
    return seq(
      field('keyword', token(prec(1, keyword))),
      field('value', $._refineable_value_expression),
    );
  } else {
    return seq(
      field('keyword', token(prec(1, keyword))),
      field('value', $._fixed_value_expression),
    );
  }
}
