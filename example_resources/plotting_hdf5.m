clear
close all
%%
filename = "output.h5";
h5disp(filename);
N_pass = h5readatt(filename, "/", "N_pass");
N_frames = h5readatt(filename, "/", "N_frames");

% transpose is needed since c++ is row-major but matlab is column major
pass=N_pass-1;
passGroup = sprintf("/Pass_%d", pass);
X = h5read(filename, sprintf("%s/X", passGroup))';
Y = h5read(filename, sprintf("%s/Y", passGroup))';
U = h5read(filename, sprintf("%s/U/frame%03d", passGroup, N_frames-1))';
V = h5read(filename, sprintf("%s/V/frame%03d", passGroup, N_frames-1))';


skp=5;
scaler=5;
figure
set(gcf,'Position',[50,50,800,800])
im = imread("cam1_im_000_A_adjusted.tiff"); colormap gray;
imagesc(flip(im,1))
set(gca, 'YDir', 'normal');
hold on
quiver(X(1:skp:end,1:skp:end), ...
    Y(1:skp:end,1:skp:end), ...
    scaler*U(1:skp:end,1:skp:end), ...
    scaler*V(1:skp:end,1:skp:end), 'AutoScale','off');
hold off
axis equal

% figure
% plot(U(:,50), Y(:,50))
%% converting uint16 to uint8

im = imread("cam1_im_000_A.tiff");
im = im2uint8(im);
im = imadjust(im);

imwrite(im,"cam1_im_000_A_adjusted.tiff")
im = imread("cam1_im_000_B.tiff");
im = im2uint8(im);
im = imadjust(im);
imwrite(im,"cam1_im_000_B_adjusted.tiff")

% % window = ones(16);
% % F = fft2(window);
% % F=abs(F).^2;
% % F=ifft2(F);
% % contourf(F)
% 
% % Create a uniform square window (all ones)
% windowSize=16;
% fftSize = 2*windowSize;
% window = ones(windowSize, windowSize);
% 
% % Zero-pad the window to the FFT size
% window_padded = padarray(window, [(fftSize-windowSize)/2, (fftSize-windowSize)/2], 0, 'both');
% 
% % Compute the FFT of the padded window
% fft_window = fft2(window_padded);
% 
% % The autocorrelation is the magnitude squared of the FFT
% % This is the weighting function in the frequency domain
% fft_autocorr = abs(fft_window).^2;
% 
% % Take the inverse FFT to get the weighting function in the spatial domain
% weighting_function = fftshift(ifft2(fft_autocorr));
% 
% contourf(weighting_function)