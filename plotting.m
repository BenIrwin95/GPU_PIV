clear
close all
%%
data = unpackData("vec_000.dat");

pass=1;
figure
quiver(data.X{pass},data.Y{pass},data.U{pass}, data.V{pass})
axis equal
% 
% 
% ##
% ##%%
% ##%% PIV synthetic data
% ##
% ##% create flowfield
% ##[X,Y] = ndgrid(linspace(-1,1,100), linspace(-1,1,100));
% ##
% ##U_mag = 0.01*clip(1./(X.^2 + Y.^ 2),-1,1);
% ##theta = atan2(Y,X);
% ##U=U_mag.*sin(theta);
% ##V=-U_mag.*cos(theta);
% ##
% ##U_interp = scatteredInterpolant(X(:),Y(:), U(:));
% ##V_interp = scatteredInterpolant(X(:),Y(:), V(:));
% ##
% ##
% ##
% ##%n_part=100000;
% ##n_part=10000;
% ##x_part1 = -1 + 2*rand(n_part,1);
% ##y_part1 = -1 + 2*rand(n_part,1);
% ##
% ##dt=1;
% ##
% ##
% ##x_part2 = x_part1 + dt*U_interp(x_part1, y_part1);
% ##y_part2 = y_part1 + dt*V_interp(x_part1, y_part1);
% ##
% ##figure
% ##hold on
% ##quiver(X,Y,U,V)
% ##plot(x_part1,y_part1, 'ko')
% ##plot(x_part2,y_part2, 'ro')
% ##hold off
% ##axis equal
% ##
% ##
% ##% image creation
% ##stencil = [0.25, 0.25, 0.25, 0.25, 0.25;
% ##           0.25, 0.50, 0.50, 0.50, 0.25;
% ##           0.25, 0.50, 1.00, 0.50, 0.25;
% ##           0.25, 0.50, 0.50, 0.50, 0.25;
% ##           0.25, 0.25, 0.25, 0.25, 0.25];
% ##% n=11;
% ##% stencil = zeros(n);
% ##% for i=1:n
% ##%     for j=1:n
% ##%         stencil(i,j) = 1 - (1/100)*((i-ceil(n/2))^2 + (j-ceil(n/2))^2);
% ##%     end
% ##% end
% ##
% ##
% ##window_halfsize = floor(size(stencil,1)/2);
% ##
% ##% 2560x1600
% ##N_image = 2023;
% ##m_per_pix = (max(X(:))-min(X(:)))/N_image;
% ##I1 = zeros(N_image);
% ##for i=1:length(x_part1)
% ##    try
% ##        i_image = floor(interp1(linspace(-1,1, size(I1,1)), 1:size(I1,1), x_part1(i)));
% ##        j_image = floor(interp1(linspace(-1,1, size(I1,2)), 1:size(I1,1), y_part1(i)));
% ##        i_range=i_image-window_halfsize:i_image+window_halfsize;
% ##        j_range=j_image-window_halfsize:j_image+window_halfsize;
% ##        I1(i_range,j_range) = I1(i_range,j_range) + stencil;
% ##    catch
% ##    end
% ##end
% ##I2 = zeros(N_image);
% ##for i=1:length(x_part1)
% ##    try
% ##    i_image = floor(interp1(linspace(-1,1, size(I1,1)), 1:size(I1,1), x_part2(i)));
% ##    j_image = floor(interp1(linspace(-1,1, size(I1,2)), 1:size(I1,1), y_part2(i)));
% ##    i_range=i_image-window_halfsize:i_image+window_halfsize;
% ##    j_range=j_image-window_halfsize:j_image+window_halfsize;
% ##    I2(i_range,j_range) = I2(i_range,j_range) + stencil;
% ##    catch
% ##    end
% ##end
% ##clear i_image j_image i_range j_range x_part1 x_part2 y_part1 y_part2 stencil U_mag theta U_interp V_interp
% ##imshowpair(I1,I2)
% ##
% ##window = 33; % needs to be odd
% ##%window=65;
% ##window_halfsize = floor(window/2);
% ##overlap = 0.5;
% ##expected_pix_shift = 12;
% ##window2 = window+2*expected_pix_shift;
% ##
% ##im1 = I1;
% ##im2 = I2;
% ##
% ##save("true_data", 'X', 'Y', 'U', 'V', 'm_per_pix')
% ##imwrite(uint8(im1*255), "im1.tiff")
% ##imwrite(uint8(im2*255), "im2.tiff")
