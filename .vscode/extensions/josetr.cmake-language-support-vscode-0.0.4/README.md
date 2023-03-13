# CMake Language Support

CMake Language Support implements all of your favorite code editor features and is currently available for Visual Studio Code. An interactive preview is available at https://josetr.github.io/cmake-language-support.

Please note that we do our best to get most of the command signatures and parameters by scanning the whole CMake Documentation but it's currently impossible to be 100% accurate due to the nature of the documentation.

[![CI](https://josetr.github.io/cmake-language-support/cmake-vscode.png)](https://github.com/josetr/cmake-language-support/blob/HEAD/cmake-vscode.png)

## Features

* Syntax Highlighting
* Auto Completion
* Quick Info
* Signature Help
* Syntax Errors / Warnings
* Format Document
* Format Selection
* Rename Symbol
* Comment Line / Block
* Document Symbols
* Folding Ranges
* Find All References
* Go to Definition
* Document Highlights
* Brace Completion
* Brace Matching
* Brace Surrounding
* Smart Indentation

### FileAPI (Experimental)

Description: Enables experimental use of https://cmake.org/cmake/help/git-stage/manual/cmake-file-api.7.html in order to provide more accurate results.

Requirements: The `CMake Tools` extension must be installed and the `cmakeLanguageSupport.enableFileAPI` setting must be set to `true`.

Every time `CMake Tools` configures the cmake project (usually when you save a CMakeLists.txt file), our `cmake-file-api` query will be executed and we will be able to:

1) Provide accurate results of:
	* Targets
	* Configurations
	* Toolchains

2) Get the exact cmake files used by the cmake project and parse them all in order to:
	* Search user-declared functions, macros, and variables in every file
	* Enable the following features to work across all of the available files
		* Find All References
		* Go To Definition
		* Workspace Symbols
		* Auto Completion
		* Signature Help
		* Quick Info