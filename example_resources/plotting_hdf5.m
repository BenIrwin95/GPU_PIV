clear
close all
%%
filename = "output.h5";
h5disp(filename);
N_pass = h5readatt(filename, "/", "N_pass");
N_frames = h5readatt(filename, "/", "N_frames");

% transpose is needed since c++ is row-major but matlab is column major
passGroup = sprintf("/Pass_%d", N_pass-1);
X = h5read(filename, sprintf("%s/X", passGroup))';
Y = h5read(filename, sprintf("%s/Y", passGroup))';
U = h5read(filename, sprintf("%s/U/frame%03d", passGroup, N_frames-1))';
V = h5read(filename, sprintf("%s/V/frame%03d", passGroup, N_frames-1))';


skp=3;
scaler=3;
figure
set(gcf,'Position',[50,50,800,800])
quiver(X(1:skp:end,1:skp:end), ...
    Y(1:skp:end,1:skp:end), ...
    scaler*U(1:skp:end,1:skp:end), ...
    scaler*V(1:skp:end,1:skp:end), 'AutoScale','off');
axis equal

figure
plot(U(:,50), Y(:,50))