
set PYO3_PYTHON=C:\Users\Zach\AppData\Local\Programs\Python\Python310\python.exe

cargo build --release

copy target\release\rust_evogression.dll ..\evogression\rust_evogression.pyd
