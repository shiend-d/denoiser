{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [ (pkgs.python314.withPackages (ps: with ps; [ numpy opencv4 pillow ])) ];
          shellHook = ''export PYTHONPATH="$PWD:$PYTHONPATH"; echo "Python distributed-upscaler devShell ready"'';
        };
      });
    };
}
