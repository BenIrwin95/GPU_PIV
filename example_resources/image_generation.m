clear
close all
%n_part=100000;
n_part=60000;
x_part1 = -1 + 2*rand(n_part,1);
y_part1 = -1 + 2*rand(n_part,1);




% image creation
stencil = [0.25, 0.25, 0.25, 0.25, 0.25;
           0.25, 0.50, 0.50, 0.50, 0.25;
           0.25, 0.50, 1.00, 0.50, 0.25;
           0.25, 0.50, 0.50, 0.50, 0.25;
           0.25, 0.25, 0.25, 0.25, 0.25];



window_halfsize = floor(size(stencil,1)/2);

% 2560x1600
N_image = 2023;
I1 = zeros(N_image);
for i=1:length(x_part1)
    try
        i_image = floor(interp1(linspace(-1,1, size(I1,1)), 1:size(I1,1), x_part1(i)));
        j_image = floor(interp1(linspace(-1,1, size(I1,2)), 1:size(I1,1), y_part1(i)));
        i_range=i_image-window_halfsize:i_image+window_halfsize;
        j_range=j_image-window_halfsize:j_image+window_halfsize;
        I1(i_range,j_range) = I1(i_range,j_range) + stencil;
    catch
    end
end

I2 = zeros(size(I1));
shiftVal_x = -8;
shiftVal_y = 0;
for i=1:size(I2,1)
  for j=1:size(I2,2)
    idx_i = i+shiftVal_y;
    idx_j = j+shiftVal_x;
    if(idx_i >= size(I2,1) || idx_i < 1 || idx_j >=size(I2,2) || idx_j < 1)
      I2(i,j)=0;
    else
      I2(i,j) = I1(idx_i, idx_j);
    end
  end
end




imwrite(uint8(I1*255), "im1.tiff")
imwrite(uint8(I2*255), "im2.tiff")
