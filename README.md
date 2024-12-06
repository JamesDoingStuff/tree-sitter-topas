# Tree-sitter TOPAS

A [TOPAS-Academic](http://www.topas-academic.net/) grammar for [Tree-sitter](https://github.com/tree-sitter/tree-sitter).

## Usage

### Neovim

This grammar can be setup within [Neovim](https://github.com/neovim/neovim/) to provide highlighting for `.inp` files. 
It requires:
- **Neovim 0.10** or newer.
- [Nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter) 
- Git
- A C compiler
  
1. Add the following Lua snippet to your `init.vim` or `init.lua`:
    ```lua
    vim.filetype.add {
      extension = {
        inp = 'topas',
      }
    }

    local parser_config = require "nvim-treesitter.parsers".get_parser_configs()
    parser_config.topas = {
      install_info = {
        url = "https://github.com/JamesDoingStuff/tree-sitter-topas", -- can be replaced with path to cloned repo
        files = {"src/parser.c"}, 
        branch = "main",
        requires_generate_from_grammar = false,
      },
      filetype = "inp", 
    }

    require'nvim-treesitter.configs'.setup{
      highlight={
        enable = true,
        additional_vim_regex_highlighting = false,
      },
    }
    ```
2. To apply highlights to a file, queries are used to match up nodes of the syntax tree, such as `(identifier)`, to a pre-configured higlight group, such as `variable`.

   The highlights file cannot yet be installed automatically;
   download [highlights.scm](./queries/highlights.scm) from this repository and add it to a location within your
   Neovim runtime path inside a directory named `queries/topas/` e.g., `.config/nvim/queries/topas/highlights.scm`

3. Open Neovim and run `:TSInstall topas`. Once the installation has completed, highlighting should be active for TOPAS. Check by running `:checkhealth nvim-treesitter` and looking for a tick
   in the first column of the row labelled `topas`.

#
### VS Code

Through a third-party extension, this grammar can be used to provide semantic highlighting for TOPAS `.inp` files in Visual Studio Code.

**Requirements:**
- [VS Code](https://code.visualstudio.com/download)
- The [tree-sitter-vscode](https://marketplace.visualstudio.com/items?itemName=AlecGhost.tree-sitter-vscode) extension

1. You will also need local copies of `tree-sitter-topas.wasm` and `highlights.scm`, included in [each release](https://github.com/JamesDoingStuff/tree-sitter-topas/releases).
   Experimental versions of the .wasm file can also be sourced as artifacts of the [CI workflow](https://github.com/JamesDoingStuff/tree-sitter-topas/actions/workflows/ci.yml).
   
2. In the `package.json` for the tree-sitter-vscode extension, add the following key to the "contributes" section:

   ```JSON
    "languages": [
      {
        "id": "topas",
        "extensions": [
          ".inp"
        ]
      }
    ]
    ```

    If you are having trouble locating the `package.json`, bring up the command palette (`Ctrl`+`shift`+`P`), use the command 
    `> Extensions: Open extensions folder`, then look in the folder `alecghost.tree-sitter-vscode-0.0.1`
   
3. In that same file, change the `activationEvents` array to the following:
   
   ```JSON
   "activationEvents": [
      "onLanguage:topas"
    ]
   ```

4. In your `settings.json` (`> Open user settings (JSON)`), add the following, updating the file paths to point to the respective files:
   
    ```JSON
     "tree-sitter-vscode.languageConfigs": [
       {
         "lang": "topas",
         "parser": "/path/to/tree-sitter-topas.wasm",
         "highlights": "/path/to/highlights.scm"
       }
     ]
    ```

5. Now, a `.inp` file should register as having a language mode of 'topas' in the lower right corner of VS Code. To turn on highlighting, open settings (`Ctrl`+`,`), search for 'semantic highlighting'
   and make sure `Editor > Semantic Higlighting` is enabled.


## Features

- [x] Comments
- [x] Numbers
- [x] Strings
- [x] Macro declarations
- [x] Macro invocations
- [x] Preprocessor directives
- [x] Unary operators
- [x] Binary operators
- [x] Equations
- [x] Refinement signallers
- [x] Variable declaration keywords (`prm`, `local`)
- [x] `existing_prm` keyword
- [x] `site` declaration keywords
- [x] Unstructured keyword identification
- [ ] Built-in maths functions 
- [ ] Structured keywords
- [ ] File path strings
- [ ] Data blocks { ... }
- [ ] Function declarations (`fn` keyword)
- [ ] If statements (`if` keyword)
