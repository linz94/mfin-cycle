% load data
clc
clear

if isunix
    sep='/';
else
    sep = '\';
end

addpath('examples')

load('examples/imgs.mat')
load('mfin_cycle_results.mat')

% visualization
dx=4;
[X,Y] = meshgrid(1:dx:224,1:dx:224);

for i = 1:20

u_im1 = squeeze(p1(i,:,:,1));
v_im1 = squeeze(p1(i,:,:,2));

u_im2 = squeeze(p2(i,:,:,1));
v_im2 = squeeze(p2(i,:,:,2));

subU2 = u_im2(1:dx:end,1:dx:end);
subV2 = v_im2(1:dx:end,1:dx:end);

subU1 = u_im1(1:dx:end,1:dx:end);
subV1 = v_im1(1:dx:end,1:dx:end);

figure(1)

subplot(2,3,1)
imagesc(refs(:,:,(i+1)*2)',[0,255]);
colormap gray;
axis tight
axis equal
axis off
title('Ground Truth')

subplot(2,3,2)
imagesc(squeeze(x3_hat(i,:,:)),[0,255]);
axis tight
axis equal
axis off
title('Interpolated Images')

subplot(2,3,3)
imagesc(squeeze(x3_hat(i,:,:))-refs(:,:,(i+1)*2)',[-50,50]);
axis tight
axis equal
axis off
title('Difference Images')

subplot(2,3,4)
imagesc(refs(:,:,(i+1)*2-1)'-refs(:,:,(i+1)*2)', [-50, 50]);
hold on
quiver(X,Y,subU1,subV1,0, 'r')
hold off
title('Displacement field (t->t-1)')
axis equal
axis tight
axis off

subplot(2,3,5)
imagesc(refs(:,:,(i+1)*2+1)'-refs(:,:,(i+1)*2)', [-50, 50]);
hold on
quiver(X,Y,subU2,subV2,0, 'r')
hold off
title('Displacement field (t+1->t)')
axis equal
axis tight
axis off

pause(0.1)

end