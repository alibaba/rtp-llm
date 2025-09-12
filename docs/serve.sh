# Clean and serve documentation with auto-build
make clean
make compile
make html
PORT=20880 make serve
