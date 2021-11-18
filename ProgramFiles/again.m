% we want to compare the difference between the area integral of
% different paths on a torus

s = sysf_three_link_lowRe_CoM();
n=30;
alphas = linspace(-pi,pi,n);
[A1,A2]=ndgrid(alphas,alphas);
s.grid.eval = cell(2,1);
s.grid.eval{1} = A1;
s.grid.eval{2} = A2;
s.grid.finite_element = s.grid.eval;
At = zeros(n,n,3,2);
for i = 1:n
    for j = 1:n
                A = s.A(alphas(i),alphas(j));
                At(i,j,:,:) = -A;
    end
end
for m = 1:3
    for n = 1:2
s.vecfield.eval.content.Avec{m,n} = squeeze(At(:,:,m,n));
    end
end
s.vecfield.finite_element.content.Avec = s.vecfield.eval.content.Avec;
s.vecfield.eval.content.Avec_optimized = s.vecfield.eval.content.Avec;
s.singularity = 0;
s = calc_constraint_curvature(s);

%%
surf(A1,A2,s.DA{3})
axis equal
view(2)
shading interp
title("curl direction \theta")

%% draw torus

set(gcf,'units','points','position',[0,0,1300,500])

subplot(2,2,1)
R = 3;
r = 1;
th = 0:pi/10:2*pi;
ph = 0:pi/10:2*pi;
[TH,PH] = ndgrid(th,ph);
x = (R+r*cos(TH)).*cos(PH);
y = (R+r*cos(TH)).*sin(PH);
z = r*sin(TH);
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
title('tangent surface')


X1 = sin(PH);
X2 = -cos(PH);
X3 = zeros(size(PH));
Y1 = cos(PH).*sin(TH);
Y2 = sin(PH).*sin(TH);
Y3 = -cos(TH);
quiver3(x,y,z,X1,X2,X3,'b');
hold on
quiver3(x,y,z,Y1,Y2,Y3,'b');
hold on

subplot(2,2,2)
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
title('tangent covector field')

% Ax = zeros(size(PH));
% Ay = zeros(size(PH));
% for i = 1:length(th)
%     for j = 1:length(ph)
%         A = s.A(th(i),ph(j));
%         Ax(i,j) = A(1,1);
%         Ay(i,j) = A(1,2);
%     end
% end
% V1 = Ax.*X1+Ay.*Y1;
% V2 = Ax.*X2+Ay.*Y2;
% V3 = Ax.*X3+Ay.*Y3;
V1 = zeros(size(PH));
V2 = zeros(size(PH));
V3 = zeros(size(PH));
for i = 1:length(th)
    for j = 1:length(ph)
        A = s.A(th(i),ph(j));
        theta = th(i);
        phi = ph(j);
        vec = [-r*cos(phi)*sin(theta) -r*sin(phi)*sin(theta) r*cos(theta); -(R+r*cos(theta))*sin(phi) (R+r*cos(theta))*cos(phi) 0]\[A(1,1);A(1,2)];
%         [-r*cos(phi)*sin(theta) -r*sin(phi)*sin(theta) r*cos(theta); -(R+r*cos(theta))*sin(phi) (R+r*cos(theta))*cos(phi) 0]*vec - [A(1,1);A(1,2)]
        vec = vec - dot(vec,cross([X1(i,j),X2(i,j),X3(i,j)],[Y1(i,j),Y2(i,j),Y3(i,j)]));
%         norm(cross([X1(i,j),X2(i,j),X3(i,j)],[Y1(i,j),Y2(i,j),Y3(i,j)]))
        V1(i,j) = vec(1);
        V2(i,j) = vec(2);
        V3(i,j) = vec(3);
    end
end
quiver3(x,y,z,V1,V2,V3,'r','LineWidth',2)

subplot(2,2,3)
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
quiver3(x,y,z,X1,X2,X3,'b');
hold on
quiver3(x,y,z,Y1,Y2,Y3,'b');
hold on
quiver3(x,y,z,V1,V2,V3,'r','LineWidth',2)
hold on
title('tangent covector field')

subplot(2,2,4)
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
[X,Y,Z] = ndgrid(-4:4,-4:4,-1:1);
v1 = griddata(x,y,z,V1,X,Y,Z);
v2 = griddata(x,y,z,V2,X,Y,Z);
v3 = griddata(x,y,z,V3,X,Y,Z);
quiver3(X,Y,Z,v1,v2,v3,'r','LineWidth',2);
hold on
title('interpolated covector field')

%%
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
[X,Y,Z] = meshgrid(-4:0.5:4,-4:0.5:4,-1:0.5:1);
v1 = griddata(x,y,z,V1,X,Y,Z);
v2 = griddata(x,y,z,V2,X,Y,Z);
v3 = griddata(x,y,z,V3,X,Y,Z);
quiver3(X,Y,Z,v1,v2,v3,'r');
hold on
title('interpolated covector field')

v1(isnan(v1)) = 0;
v2(isnan(v2)) = 0;
v3(isnan(v3)) = 0;

%%

s.A3x = @(x,y,z) [interp3(X,Y,Z,v1,x,y,z), interp3(X,Y,Z,v2,x,y,z), interp3(X,Y,Z,v3,x,y,z)];
% test_a1 = alphas;
% test_a2 = ones(1,length(alphas)) * alphas(17);
% test_a1 = zeros(1,length(th));
% test_a2 = th;
test_a1 = th;
test_a2 = th;
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
plot3(test_x,test_y,test_z,'r')

%%
R2=0;R3=0;
for i = 1:length(test_a1)-1
    A = s.A(test_a1(i),test_a2(i));
    R2 = R2 + A(1,:)*[test_a1(i+1)-test_a1(i);test_a2(i+1)-test_a2(i)];
    R3 = R3 + s.A3x(test_x(i),test_y(i),test_z(i)) * [test_x(i+1)-test_x(i);test_y(i+1)-test_y(i);test_z(i+1)-test_z(i)];
end

R2
R3

%%

p.phi_def{1} = @(t) t;
p.phi_def{2} = @(t) t;

sol = ode45(@(t,y) helper_function(t,y,s,p),[-pi,pi],[0 0 0]');
disp = deval(sol,pi);

%%
load('/Users/ruijief/Desktop/Kylin/cc/sysplotter_fric/ProgramFiles/sys_draw_fcns/colorsets/BlackWhiteRedColormap.mat','blackwhitered');
CM = blackwhitered;

[X,Y] = ndgrid(alpha,alpha);

set(gcf,'units','points','position',[0,0,1300,500])

subplot(1,2,1)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi,-pi],[-pi,pi,pi,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);
subplot(1,2,2)
title('cBVI with different trajectories')
hold on
xlabel('trajectory', 'fontsize', 15)
ylabel('displacement', 'fontsize', 15)
hold on
plot([1,2,3,4],[4.4131e-4, 3.1471e-4, 0.5874e-4, 6.3061e-16])

pause(2)

subplot(1,2,1)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi,-pi],[-pi,pi,pi,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);
subplot(1,2,2)
title('cBVI with different trajectories')
hold on
xlabel('trajectory', 'fontsize', 15)
ylabel('displacement', 'fontsize', 15)
hold on
plot([1,2,3,4],[4.4131e-4, 3.1471e-4, 0.5874e-4, 6.3061e-16])

pause(2)

subplot(1,2,1)
hold off
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi,-pi],[-pi,pi,pi,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);
subplot(1,2,2)
title('cBVI with different trajectories')
hold on
xlabel('trajectory', 'fontsize', 15)
ylabel('displacement', 'fontsize', 15)
hold on
plot([1,2,3,4],[4.4131e-4, 3.1471e-4, 0.5874e-4, 6.3061e-16])

pause(2)

subplot(1,2,1)
hold off
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi,-pi],[-pi,pi,pi,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);
subplot(1,2,2)
title('cBVI with different trajectories')
hold on
xlabel('trajectory', 'fontsize', 15)
ylabel('displacement', 'fontsize', 15)
hold on
plot([1,2,3,4],[4.4131e-4, 3.1471e-4, 0.5874e-4, 6.3061e-16])


%%

load('/Users/ruijief/Desktop/Kylin/cc/sysplotter_fric/ProgramFiles/sys_draw_fcns/colorsets/BlackWhiteRedColormap.mat','blackwhitered');
CM = blackwhitered;

[X,Y] = ndgrid(alpha,alpha);

set(gcf,'units','points','position',[0,0,1300,500])
subplot(2,4,1)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi,-pi],[-pi,pi,pi,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);

subplot(2,4,2)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi*2/3,-pi],[-pi,pi,pi*2/3,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);

subplot(2,4,3)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi,-pi/3,-pi],[-pi,pi,pi/3,-pi],[100,100,100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);

subplot(2,4,4)
title('cBVI calculation')
hold on
xlabel('\alpha_1', 'fontsize', 15)
ylabel('\alpha_2', 'fontsize', 15)
hold on
surf(X,Y,s.DA{1})
colormap(CM)
shading interp
axis equal
hold on
colorbar;
plot3([-pi,pi],[-pi,pi],[100,100],'b')
xlim([-pi pi])
ylim([-pi pi])
view(2);

subplot(2,4,5)
title('cBVI with different trajectories')
hold on
xlabel('trajectory', 'fontsize', 15)
ylabel('displacement', 'fontsize', 15)
hold on
plot([1,2,3,4],[4.4131e-4, 3.1471e-4, 0.5874e-4, 6.3061e-16])




function dX = helper_function(t,X,s,gait)

	% X is the accrued displacement and cost

	[xi] = s.A(gait.phi_def{1}(t),gait.phi_def{2}(t)) * [1;1];
		
	% Rotate body velocity into world frame
	theta = X(3);
    v = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]*xi;
		
	% Combine the output
	dX = v ;
    dX = xi;
	

end