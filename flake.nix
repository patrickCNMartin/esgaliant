{
  description = "Esgaliant - High-performance ML/biological simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;  # For CUDA if needed
          };
        };

        python = pkgs.python311;

        # System dependencies for compiled extensions
        systemDeps = with pkgs; [
          # Build tools
          gcc
          gfortran
          cmake
          pkg-config

          # Linear algebra (for NumPy/SciPy)
          openblas
          lapack

          # Optional: CUDA support
          # cudaPackages.cudatoolkit
          # cudaPackages.cudnn

          # Optional: For visualization/GUI
          # libGL
          # libGLU
        ];

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            uv
            git

            # Performance profiling tools
            # linuxPackages.perf  # Linux only - uncomment if on Linux
            # valgrind            # Uncomment for memory debugging

            # Documentation
            graphviz  # For diagrams

          ] ++ systemDeps;

          shellHook = ''
            echo "Esgaliant Development Environment"
            echo "===================================="

            # Set up environment variables for compiled extensions
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemDeps}:$LD_LIBRARY_PATH"
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPath "lib/pkgconfig" systemDeps}:$PKG_CONFIG_PATH"

            # JAX configuration
            export JAX_ENABLE_X64=1
            export JAX_PLATFORMS=cpu  # Change to 'cuda' if GPU available

            # NumPy/OpenBLAS optimization
            export OPENBLAS_NUM_THREADS=1  # Let JAX/PyTorch handle threading
            export MKL_NUM_THREADS=1

            # Python optimization
            export PYTHONHASHSEED=0  # Reproducible hash seeds
            export PYTHONPATH=$PWD/src:$PYTHONPATH

            # Create/activate venv
            if [ ! -d .venv ]; then
              echo "Creating virtual environment with uv..."
              uv venv --python ${python}/bin/python
              echo "Created .venv"
            fi

            source .venv/bin/activate

            # Gitignore management
            if ! grep -q ".venv/" .gitignore 2>/dev/null; then
              echo ".venv/" >> .gitignore
              echo "Added .venv/ to .gitignore"
            fi

            if ! grep -q "__pycache__/" .gitignore 2>/dev/null; then
              cat >> .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.prof
*.lprof
*.egg-info/
dist/
build/
.DS_Store
EOF
              echo "Added Python artifacts to .gitignore"
            fi

            # Check if pyproject.toml exists
            if [ ! -f pyproject.toml ]; then
              echo "No pyproject.toml found!"
              echo "Please use the comprehensive pyproject.toml provided."
              exit 1
            fi

            # Install dependencies
            echo "Installing dependencies with uv..."
            uv pip install -e '.[all]'

            # Install pre-commit hooks if config exists
            if [ -f .pre-commit-config.yaml ]; then
              if command -v pre-commit &> /dev/null; then
                pre-commit install
                echo "Pre-commit hooks installed"
              fi
            else
              echo " Tip: Add .pre-commit-config.yaml for automated linting"
            fi

            # Pre-compile JAX (warmup)
            echo "Warming up JAX..."
            python -c "import jax; import jax.numpy as jnp; jnp.ones(1).block_until_ready()" 2>/dev/null && echo "JAX ready" || echo "JAX not installed yet"

            # Verify installation
            if python -c "import esgaliant" 2>/dev/null; then
              ESGALIANT_VERSION=$(python -c "import esgaliant; print(esgaliant.__version__)" 2>/dev/null || echo "unknown")
              echo "esgaliant v$ESGALIANT_VERSION installed"
            else
              echo "esgaliant not yet importable"
              echo "Create src/esgaliant/__init__.py with: __version__ = '0.1.0'"
            fi

            echo ""
            echo " Environment ready!"
            echo ""
            echo "Quick commands:"
            echo "  ruff format src/             # Format code"
            echo "  ruff check src/              # Lint code"
            echo "  mypy src/                    # Type check"
            echo "  pytest tests/                # Run tests"
            echo "  pytest --cov=src             # Test coverage"
            echo ""
            echo "Performance:"
            echo "  pytest tests/benchmarks/ --benchmark-only"
            echo "  python -m scalene your_script.py"
            echo ""
          '';
        };

        # Optional: Package the project
        packages.default = python.pkgs.buildPythonPackage {
          pname = "esgaliant";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            scipy
            jax
            numba
            torch
            polars
            # Add other runtime deps available in nixpkgs
          ];

          buildInputs = systemDeps;

          # Skip tests during build
          doCheck = false;
        };
      }
    );
}
