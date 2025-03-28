let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = [
    
    pkgs.rustup

    pkgs.python3
    pkgs.python3Packages.matplotlib
  
  ];
}
