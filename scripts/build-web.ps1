# Build the WebAssembly module and generate its JavaScript glue into web/pkg.
# Run from the project root:   .\scripts\build-web.ps1
$ErrorActionPreference = "Stop"
$env:Path += ";$env:USERPROFILE\.cargo\bin"

cargo build -p tlm-wasm --target wasm32-unknown-unknown --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

# --target web produces an ES module that works in a browser without a bundler.
wasm-bindgen --target web --out-dir web/pkg --out-name tlm_wasm `
    target/wasm32-unknown-unknown/release/tlm_wasm.wasm
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed (is its version exactly 0.2.130?)" }

Get-ChildItem web/pkg | Select-Object Name, Length
Write-Host "`nDone. Serve it with:  python -m http.server 8080 --directory web"
