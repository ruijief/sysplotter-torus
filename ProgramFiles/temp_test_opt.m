s = sysf_three_link_lowRe_CoM();
% s = sysf_three_link_kinematic();
s.A2 = s.A;

R = 3;
r = 1;
th = 0:pi/10:2*pi;
ph = 0:pi/10:2*pi;
[TH,PH] = ndgrid(th,ph);
x = (R+r*cos(TH)).*cos(PH);
y = (R+r*cos(TH)).*sin(PH);
z = r*sin(TH);
X1 = sin(PH);
X2 = -cos(PH);
X3 = zeros(size(PH));
Y1 = cos(PH).*sin(TH);
Y2 = sin(PH).*sin(TH);
Y3 = -cos(TH);
V1 = zeros(size(PH));
V2 = zeros(size(PH));
V3 = zeros(size(PH));
for i = 1:length(th)
    for j = 1:length(ph)
% i=2;j=1;
        A = s.A(th(i),ph(j));
        theta = th(i);
        phi = ph(j);
        J = [-r*cos(phi)*sin(theta) -(R+r*cos(theta))*sin(phi);-r*sin(phi)*sin(theta) (R+r*cos(theta))*cos(phi); r*cos(theta) 0];
        vec = A/J;
        x_axis = [X1(i,j) X2(i,j) X3(i,j)];
        y_axis = [Y1(i,j) Y2(i,j) Y3(i,j)];
        z_axis = cross(x_axis,y_axis);
%         new_A = vec * (eye(3) - z_axis'*z_axis);
        new_A = vec;
%         a=A * [th(i+1)-th(i-1); ph(i+1)-ph(i-1)]
%         b=new_A * [x(i+1)-x(i-1); y(i+1)-y(i-1); z(i+1)-z(i-1)]
        V11(i,j) = new_A(1,1);
        V12(i,j) = new_A(1,2);
        V13(i,j) = new_A(1,3);
        V21(i,j) = new_A(2,1);
        V22(i,j) = new_A(2,2);
        V23(i,j) = new_A(2,3);
        V31(i,j) = new_A(3,1);
        V32(i,j) = new_A(3,2);
        V33(i,j) = new_A(3,3);
    end
end


%%
surf(x,y,z);
alpha(0.8)
axis equal
shading interp
hold on
quiver3(x,y,z,V11,V12,V13,'r','LineWidth',2)
% plot3(x(16,:),y(16,:),z(16,:),'b','LineWidth',3)

%%
Rp = 3;
rp = 2;
xp = (Rp+rp*cos(TH)).*cos(PH);
yp = (Rp+rp*cos(TH)).*sin(PH);
zp = rp*sin(TH);
surf(xp(1:6,:),yp(1:6,:),zp(1:6,:))
hold on
surf(xp(16:21,:),yp(16:21,:),zp(16:21,:))
alpha(0.3)
axis equal
shading interp
hold on
quiver3(xp(1:6,:),yp(1:6,:),zp(1:6,:),V11(1:6,:),V12(1:6,:),V13(1:6,:),'b','LineWidth',1)
quiver3(xp(16:21,:),yp(16:21,:),zp(16:21,:),V11(16:21,:),V12(16:21,:),V13(16:21,:),'b','LineWidth',1)

%%

[X,Y,Z] = ndgrid(-4:0.1:4,-4:0.1:4,-1:0.1:1);
% v11 = griddata(x,y,z,V11,X,Y,Z);
% v21 = griddata(x,y,z,V21,X,Y,Z);
% v31 = griddata(x,y,z,V31,X,Y,Z);
% v12 = griddata(x,y,z,V12,X,Y,Z);
% v22 = griddata(x,y,z,V22,X,Y,Z);
% v32 = griddata(x,y,z,V32,X,Y,Z);
% v13 = griddata(x,y,z,V13,X,Y,Z);
% v23 = griddata(x,y,z,V23,X,Y,Z);
% v33 = griddata(x,y,z,V33,X,Y,Z);
v11 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V11;V11(16:21,:);V11(1:6,:)],X,Y,Z);
v21 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V21;V21(16:21,:);V21(1:6,:)],X,Y,Z);
v31 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V31;V31(16:21,:);V31(1:6,:)],X,Y,Z);
v12 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V12;V12(16:21,:);V12(1:6,:)],X,Y,Z);
v22 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V22;V22(16:21,:);V22(1:6,:)],X,Y,Z);
v32 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V32;V32(16:21,:);V32(1:6,:)],X,Y,Z);
v13 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V13;V13(16:21,:);V13(1:6,:)],X,Y,Z);
v23 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V23;V23(16:21,:);V23(1:6,:)],X,Y,Z);
v33 = griddata([x;xp(16:21,:);xp(1:6,:)],[y;yp(16:21,:);yp(1:6,:)],[z;zp(16:21,:);zp(1:6,:)],[V33;V33(16:21,:);V33(1:6,:)],X,Y,Z);
v11(isnan(v11))=0;
v12(isnan(v12))=0;
v13(isnan(v13))=0;
v21(isnan(v21))=0;
v22(isnan(v22))=0;
v23(isnan(v23))=0;
v31(isnan(v31))=0;
v32(isnan(v32))=0;
v33(isnan(v33))=0;

[X,Y,Z] = meshgrid(-4:0.1:4,-4:0.1:4,-1:0.1:1);

s.A = @(a1,a2,a3) [interp3(X,Y,Z,v11,a2,a1,a3,'linear') interp3(X,Y,Z,v12,a2,a1,a3) interp3(X,Y,Z,v13,a2,a1,a3,'linear');interp3(X,Y,Z,v21,a2,a1,a3,'linear') interp3(X,Y,Z,v22,a2,a1,a3,'linear') interp3(X,Y,Z,v23,a2,a1,a3,'linear');interp3(X,Y,Z,v31,a2,a1,a3,'linear') interp3(X,Y,Z,v32,a2,a1,a3,'linear') interp3(X,Y,Z,v33,a2,a1,a3,'linear')];

%%

theta = th(5);
phi = th(1);
J = [-r*cos(phi)*sin(theta) -(R+r*cos(theta))*sin(phi);-r*sin(phi)*sin(theta) (R+r*cos(theta))*cos(phi); r*cos(theta) 0];
s.A2(theta,phi)/J
s.A((R+r*cos(theta))*cos(phi),(R+r*cos(theta))*sin(phi),r*sin(theta))

%%

test_a2 = th;
test_a1 = ones(1,length(th)) * th(5);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
%%
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
quiver3(x,y,z,V11,V12,V13,'b','LineWidth',2)
plot3(test_x,test_y,test_z,'r','LineWidth',2)

%%
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
quiver3(x,y,z,V11,V12,V13,'b','LineWidth',2)
for i=1:length(th)
    test_a1 = th;
test_a2 = ones(1,length(th)) * th(i);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
plot3(test_x,test_y,test_z,'b','LineWidth',2)
end
%%
helper11 = @(M) M(1,1);
helper12 = @(M) M(1,2);
helper13 = @(M) M(1,3);
helper21 = @(M) M(2,1);
helper22 = @(M) M(2,2);
helper23 = @(M) M(2,3);
helper31 = @(M) M(3,1);
helper32 = @(M) M(3,2);
helper33 = @(M) M(3,3);

%%
eps = 1e-6;
s.curl_x = @(a1,a2,a3) [(helper13(s.A(a1,a2+eps,a3))-helper13(s.A(a1,a2-eps,a3)))/eps/2-(helper12(s.A(a1,a2,a3+eps))-helper12(s.A(a1,a2,a3-eps)))/eps/2
    (helper11(s.A(a1,a2,a3+eps))-helper11(s.A(a1,a2,a3-eps)))/eps/2-(helper13(s.A(a1+eps,a2,a3))-helper13(s.A(a1-eps,a2,a3)))/eps/2
    (helper12(s.A(a1+eps,a2,a3))-helper12(s.A(a1-eps,a2,a3)))/eps/2-(helper11(s.A(a1,a2+eps,a3))-helper11(s.A(a1,a2-eps,a3)))/eps/2];
s.curl_y = @(a1,a2,a3) [(helper23(s.A(a1,a2+eps,a3))-helper23(s.A(a1,a2-eps,a3)))/eps/2-(helper22(s.A(a1,a2,a3+eps))-helper22(s.A(a1,a2,a3-eps)))/eps/2
    (helper21(s.A(a1,a2,a3+eps))-helper21(s.A(a1,a2,a3-eps)))/eps/2-(helper23(s.A(a1+eps,a2,a3))-helper23(s.A(a1-eps,a2,a3)))/eps/2
    (helper22(s.A(a1+eps,a2,a3))-helper22(s.A(a1-eps,a2,a3)))/eps/2-(helper21(s.A(a1,a2+eps,a3))-helper21(s.A(a1,a2-eps,a3)))/eps/2];
s.curl_th = @(a1,a2,a3) [(helper33(s.A(a1,a2+eps,a3))-helper33(s.A(a1,a2-eps,a3)))/eps/2-(helper32(s.A(a1,a2,a3+eps))-helper32(s.A(a1,a2,a3-eps)))/eps/2
    (helper31(s.A(a1,a2,a3+eps))-helper31(s.A(a1,a2,a3-eps)))/eps/2-(helper33(s.A(a1+eps,a2,a3))-helper33(s.A(a1-eps,a2,a3)))/eps/2
    (helper32(s.A(a1+eps,a2,a3))-helper32(s.A(a1-eps,a2,a3)))/eps/2-(helper31(s.A(a1,a2+eps,a3))-helper31(s.A(a1,a2-eps,a3)))/eps/2];

% diff_c
% s.curl = @(a1,a2,a3) [diff_c(helper13(s.A(a1,a2,a3)),a2)-diff_c(helper12(s.A(a1,a2,a3)),a3) diff_c(helper11(s.A(a1,a2,a3)),a3)-diff_c(helper13(s.A(a1,a2,a3)),a1) diff_c(helper12(s.A(a1,a2,a3)),a1)-diff_c(helper11(s.A(a1,a2,a3)),a2);
%     diff_c(helper23(s.A(a1,a2,a3)),a2)-diff_c(helper22(s.A(a1,a2,a3)),a3) diff_c(helper21(s.A(a1,a2,a3)),a3)-diff_c(helper23(s.A(a1,a2,a3)),a1) diff_c(helper22(s.A(a1,a2,a3)),a1)-diff_c(helper21(s.A(a1,a2,a3)),a2);
%     diff_c(helper33(s.A(a1,a2,a3)),a2)-diff_c(helper32(s.A(a1,a2,a3)),a3) diff_c(helper31(s.A(a1,a2,a3)),a3)-diff_c(helper33(s.A(a1,a2,a3)),a1) diff_c(helper32(s.A(a1,a2,a3)),a1)-diff_c(helper31(s.A(a1,a2,a3)),a2)];

%%
[X,Y,Z] = ndgrid(-4:4,-4:4,-1:0.5:1);
C1 = zeros(size(X));
C2 = zeros(size(X));
C3 = zeros(size(X));
for i=1:size(X,1)
    i
    for j = 1R+r*cos(test_a1):size(X,2)
        for k = 1:size(X,3)
        curl = s.curl_x(X(i,j,k),Y(i,j,k),Z(i,j,k));
        C1(i,j,k) = curl(1);
        C2(i,j,k) = curl(2);
        C3(i,j,k) = curl(3);
        end
    end
end


%%

results1 = [];
for tt = 1:length(th)
    test_a1 = th;
    test_a2 = ones(1,length(th)) * th(tt);
    test_x = (R+r*cos(test_a1)).*cos(test_a2);
    test_y = (R+r*cos(test_a1)).*sin(test_a2);
    test_z = r*sin(test_a1);
    
    bvi = zeros(3,1);
    bvi3 = zeros(3,1);
    for i = 1:length(test_a1)-1
        A = s.A2(test_a1(i),test_a2(i));
        A3 = s.A(test_x(i),test_y(i),test_z(i));
        theta = test_a1(i);
        phi = test_a2(i);
        J = [-r*cos(phi)*sin(theta) -(R+r*cos(theta))*sin(phi);-r*sin(phi)*sin(theta) (R+r*cos(theta))*cos(phi); r*cos(theta) 0];
        t = A*[test_a1(i+1)-test_a1(i);test_a2(i+1)-test_a2(i)];
        t3 = A3 * J * [test_a1(i+1)-test_a1(i);test_a2(i+1)-test_a2(i)];
        bvi = bvi + t;
        bvi3 = bvi3 + t3;
    end
    results1 = [results1;bvi bvi3];
end

%%

subplot(2,2,1)
title('major circle disp results')
hold on
plot(th,results(1:3:end,2))
plot(th,results(2:3:end,2))
legend('x','y')
subplot(2,2,2)
title('major circle disp results')
hold on
plot(th,results(3:3:end,2))
legend('\theta')

subplot(2,2,3)
title('minor circle disp results')
hold on
plot(th,results1(1:3:end,2))
plot(th,results1(2:3:end,2))
legend('x','y')
subplot(2,2,4)
title('minor circle disp results')
hold on
plot(th,results1(3:3:end,2))
legend('\theta')

%%

curls = {};
resultc = [];
for tt = 1:5:length(th)-1
    tt
    test_a2 = th;
    test_a1 = ones(1,length(th)) * th(tt);
    test_x = (R+r*cos(test_a1)).*cos(test_a2);
    test_y = (R+r*cos(test_a1)).*sin(test_a2);
    test_z = r*sin(test_a1);
    
    
    
    rr = R+r*cos(test_a1(1));
    plt = [];
    for i = -4:0.2:4
        i
        for j = -4:0.2:4
            if i^2+j^2<rr^2
                curlx = s.curl_x(i,j,test_z(1));
                ii = curlx(3);
                plt = [plt; i j ii];
            end
        end
    end
    
    curls{tt} = plt;
    resultc = [resultc sum(plt(:,3))];
end

%%

load('/home/ruijief/kai_ws/sysplotter-torus/ProgramFiles/sys_draw_fcns/colorsets/BlackWhiteRedColormap.mat','blackwhitered');
CM = blackwhitered;

%%

surf(x,y,z);
alpha(0.2)
axis equal
shading interp
hold on
for tt = 1:5:length(th)-1
	if tt~=1
        delete(sc3);
        delete(pl3);
    else
        pause(10);
    end
    
    test_a2 = th;
    test_a1 = ones(1,length(th)) * th(tt);
    test_x = (R+r*cos(test_a1)).*cos(test_a2);
    test_y = (R+r*cos(test_a1)).*sin(test_a2);
    test_z = r*sin(test_a1);
    pl3 = plot3(test_x,test_y,test_z,'b','LineWidth',2);
    
    
    plt = curls{tt};
    
    sc3 = scatter3(plt(:,1),plt(:,2),ones(size(plt,1),1)*r*sin(th(tt)), [], plt(:,3)/max(plt(:,3)),'filled');
    pause(5)
%     colormap(ax1,CM)
%     colorbar
end


%%

subplot(2,2,1)
surf(x,y,z);
alpha(0.2)
axis equal
shading interp
hold on
tt = 1;
title("projected curl through major circle with \alpha_2 =" + num2str(th(tt)))
test_a2 = th;
test_a1 = ones(1,length(th)) * th(tt);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
pl3 = plot3(test_x,test_y,test_z,'b','LineWidth',2);   
plt = curls{tt};   
sc3 = scatter3(plt(:,1),plt(:,2),ones(size(plt,1),1)*r*sin(th(tt)), [], plt(:,3)/max(plt(:,3)),'filled');

subplot(2,2,2)
surf(x,y,z);
alpha(0.2)
axis equal
shading interp
hold on
tt = tt+5;
title("projected curl through major circle with \alpha_2 =" + num2str(th(tt)))
test_a2 = th;
test_a1 = ones(1,length(th)) * th(tt);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
pl3 = plot3(test_x,test_y,test_z,'b','LineWidth',2);   
plt = curls{tt};   
sc3 = scatter3(plt(:,1),plt(:,2),ones(size(plt,1),1)*r*sin(th(tt)), [], plt(:,3)/max(plt(:,3)),'filled');

subplot(2,2,3)
surf(x,y,z);
alpha(0.2)
axis equal
shading interp
hold on
tt = tt+5;
title("projected curl through major circle with \alpha_2 =" + num2str(th(tt)))
test_a2 = th;
test_a1 = ones(1,length(th)) * th(tt);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
pl3 = plot3(test_x,test_y,test_z,'b','LineWidth',2);   
plt = curls{tt};   
sc3 = scatter3(plt(:,1),plt(:,2),ones(size(plt,1),1)*r*sin(th(tt)), [], plt(:,3)/max(plt(:,3)),'filled');

subplot(2,2,4)
surf(x,y,z);
alpha(0.2)
axis equal
shading interp
hold on
tt = tt+5;
title("projected curl through major circle with \alpha_2 =" + num2str(th(tt)))
test_a2 = th;
test_a1 = ones(1,length(th)) * th(tt);
test_x = (R+r*cos(test_a1)).*cos(test_a2);
test_y = (R+r*cos(test_a1)).*sin(test_a2);
test_z = r*sin(test_a1);
pl3 = plot3(test_x,test_y,test_z,'b','LineWidth',2);   
plt = curls{tt};   
sc3 = scatter3(plt(:,1),plt(:,2),ones(size(plt,1),1)*r*sin(th(tt)), [], plt(:,3)/max(plt(:,3)),'filled');




%%

c =[];
for i=1:length(test_x)
    curl = s.curl_x(test_x(i),test_y(i),test_z(i));
    c = [c curl(1)];
end

%%

surf(x,y,z);
alpha(0.3)
axis equal
shading interp
hold on
quiver3(x,y,z,V11,V12,V13,'b','LineWidth',2)
% plot3(test_x,test_y,test_z,'r','LineWidth',2)
scatter3(test_x,test_y,test_z, 30, c, '*');


%%
surf(x,y,z);
alpha(0.5)
axis equal
shading interp
hold on
quiver3(X,Y,Z,C1,C2,C3,'b','LineWidth',2)

%%
bvi = zeros(3,1);
bvi3 = zeros(3,1);
T = eye(3);
T3 = eye(3);
for i = 1:length(test_a1)-1
    A = s.A2(test_a1(i),test_a2(i));
    A3 = s.A(test_x(i),test_y(i),test_z(i));
    theta = test_a1(i);
    phi = test_a2(i);
    J = [-r*cos(phi)*sin(theta) -(R+r*cos(theta))*sin(phi);-r*sin(phi)*sin(theta) (R+r*cos(theta))*cos(phi); r*cos(theta) 0];
    t = A*[test_a1(i+1)-test_a1(i);test_a2(i+1)-test_a2(i)];
%     t3 = A3 * [test_x(i+1)-test_x(i);test_y(i+1)-test_y(i);test_z(i+1)-test_z(i)];
    t3 = A3 * J * [test_a1(i+1)-test_a1(i);test_a2(i+1)-test_a2(i)];
%     A/J
%     A3
%     xyzm = [(R+r*cos(theta)).*cos(phi),(R+r*cos(theta)).*sin(phi),r*sin(theta)]
%     xyzr = [test_x(i),test_y(i),test_z(i)]
    bvi = bvi + t;
    bvi3 = bvi3 + t3;
    T = T * [cos(t(3)) -sin(t(3)) t(1); sin(t(3)) cos(t(3)) t(2); 0 0 1];
    T3 = T3 * [cos(t3(3)) -sin(t3(3)) t3(1); sin(t3(3)) cos(t3(3)) t3(2); 0 0 1];
end

T
T3
disp = norm(T(1:2,3))
lineint = norm(T3(1:2,3))
bvi
bvi3


%%
n=50;
alpha = linspace(-pi,pi,n);
[A1,A2,A3]=ndgrid(alpha,alpha,alpha);
s.grid.eval = cell(2,1);
s.grid.eval{1} = A1;
s.grid.eval{2} = A2;
s.grid.finite_element = s.grid.eval;
At = zeros(n,n,n,3,3);
for i = 1:n
    for j = 1:n
        for k = 1:n
                A = s.A(alpha(i),alpha(j),alpha(k));
                At(i,j,k,:,:) = -A;
        end
    end
end
for m = 1:3
    for n = 1:3
s.vecfield.eval.content.Avec{m,n} = squeeze(At(:,:,:,m,n));
    end
end
s.vecfield.finite_element.content.Avec = s.vecfield.eval.content.Avec;
s.vecfield.eval.content.Avec_optimized = s.vecfield.eval.content.Avec;
s.singularity=0;
s = calc_constraint_curvature(s);


n_points = 50;
%waypoints = 
direction = 1;

y = optimalgaitgenerator_torus(s,2,n_points,waypoints,lb,ub,stretch,direction,costfunction,handles);