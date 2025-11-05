{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
        ];
        LD_LIBRARY_PATH =
          pkgs.lib.makeLibraryPath
            (with pkgs; [
              stdenv.cc.cc.lib
              libGL
              glib
              xorg.libSM
              xorg.libICE
            ]);
        shellHook = ''
          export QT_PLUGIN_PATH=${pkgs.qt5.qtbase}/lib/qt5/plugins
          # export QT_QPA_PLATFORM=xcb
        '';
      };
    };
}

