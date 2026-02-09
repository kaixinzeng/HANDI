clear;
clc;
close('all');

%% load tracked data and encoder data
load('shimmy_data/measure1_encoder.mat');
shimmy_data = readtable('shimmy_data/measure1.csv');

%% encoder side by side
f1 = create_fig(1);
plot(time, datadeg+0.75, 'DisplayName', 'Encoder', 'LineWidth', 1);
shift = 382;
plot(shimmy_data.time - shimmy_data.time(shift), shimmy_data.theta+0.4, '--', 'DisplayName', 'theta1', 'LineWidth', 1);
xlim([0, 300]);

%% plotting to clip data
create_fig(2);
plot(shimmy_data.theta+0.4, 'DisplayName', 'theta1', 'LineWidth', 1);

%% clip data
xData = {};
dt = shimmy_data.time(2) - shimmy_data.time(1);

% lco
lco_start = 4718;
lco_end = lco_start + 200;
xData{1,1} = shimmy_data.time(lco_start:lco_end)';
xData{1,1} = xData{1,1} - xData{1,1}(1);
xData{1,2} = shimmy_data.theta(lco_start:lco_end)';
xData{1,2} = xData{1,2} - mean(xData{1,2});

lco_start = 6656;
lco_end = lco_start + 200;
xData{2,1} = shimmy_data.time(lco_start:lco_end)';
xData{2,1} = xData{2,1} - xData{2,1}(1);
xData{2,2} = shimmy_data.theta(lco_start:lco_end)';
xData{2,2} = xData{2,2} - mean(xData{2,2});

% decay
decay_start = 5353;
decay_end = 5499;
xData{3,1} = shimmy_data.time(decay_start:decay_end)';
xData{3,1} = xData{3,1} - xData{3,1}(1);
xData{3,2} = shimmy_data.theta(decay_start:decay_end)';

decay_start = 2601;
decay_end = 2728;
xData{4,1} = shimmy_data.time(decay_start:decay_end)';
xData{4,1} = xData{4,1} - xData{4,1}(1);
xData{4,2} = shimmy_data.theta(decay_start:decay_end)';
xData{4,2} = xData{4,2} - xData{4,2}(end);


%% plot traj
train_traj = create_fig(3);
train_traj.Position = [10 50 700 500];
axis_list = [];

j = 1;
for i = 1:4
    set(gca, 'XTickLabel', []);
    % ax_temp = add_subplot(4,2,1+2*(j-1));
    ax_temp = add_subplot(4,1,j);
    axis_list = [axis_list, ax_temp];
    plot(xData{i,1}, xData{i,2}(1,:), 'LineWidth', 2);
    add_label('y', "$\theta$");
    ylim([-20, 20]);

    j = j + 1;
end
add_label('x', 'Time [s]', 'font_size', 25);
add_label('sgtitle', 'Training Trajectories', 'font_size', 22);
linkaxes(axis_list,'xy');

%% Delay embed
idx_train = [2,4];

SSMDim = 2;
overEmbed = 0;
[yData, opts_embd] = coordinatesEmbedding(xData, SSMDim, 'OverEmbedding', overEmbed, 'ShiftSteps', 1);

%% plotting phase space
plot_traj = [1:4];
plot_phase_dim = [1,3,5];

phase_3d_fig = create_fig();

for i = plot_traj
    x = yData{i,2}(plot_phase_dim(1),:);
    y = yData{i,2}(plot_phase_dim(2),:);
    z = yData{i,2}(plot_phase_dim(3),:);
    plot3(x, y, z, 'DisplayName', "Trajectory " + num2str(i), 'LineWidth', 3);
    scatter3(x(1), y(1), z(1), 300, 'r', 'filled', 'Displayname', 'Initial Condition');
end

legend('Location', 'northeast', 'Interpreter','latex', 'fontweight', 'bold', 'FontSize', 20);%50
xlabel('$x(t)$', 'Interpreter','latex', 'fontweight','bold', 'FontSize', 50)
ylabel('$x(t+2dt)$', 'Interpreter','latex', 'fontweight','bold', 'FontSize', 50)
zlabel('$x(t+4dt)$', 'Interpreter','latex', 'fontweight','bold', 'FontSize', 50)

%% SSMLearn
SSMDim = 2;
SSMOrder = 3;
IMInfo = IMGeometry(yData(idx_train,:), SSMDim, SSMOrder);

errorTestSet = computeParametrizationErrors(IMInfo, yData(idx_train,:))*100;
disp(errorTestSet);

Mmap = IMInfo.parametrization.map;
iMmap = IMInfo.chart.map;


%% Project onto manifold
etaData = projectTrajectories(IMInfo, yData);
state_idx = [1];
plotSSMWithTrajectories(IMInfo, state_idx, yData(idx_train,:),'Margin',20);
xlabel('$\xi_1$', 'Interpreter', 'latex','fontweight','bold', 'FontSize', 50);
ylabel('$\xi_2$', 'Interpreter', 'latex','fontweight','bold', 'FontSize', 50);
zlabel('$\theta$', 'Interpreter', 'latex','fontweight','bold', 'FontSize', 50);


%% reduced dynamics
ROMOrder = 5;

[RDInfo, ~, iTmap, ~, Tmap] = IMDynamicsFlow(etaData(idx_train,:), ...
    'R_PolyOrd', ROMOrder,'style', 'normalform', 'MaxIter',3e3, 'fig_disp_nf', 0, 'fig_disp_nfp', 0);

% advect test
[yRec, etaRec, zRec] = advect(IMInfo, RDInfo, yData);

% compute error
test_idx = [1:4];
states_for_error = [5];
fullTrajDist = computeCNMTE(yRec, yData, states_for_error); % 2nd argument needs to be original data

disp('CNMTE');
mean_CNMTE = mean(fullTrajDist(test_idx)) * 100;
disp(fullTrajDist(test_idx)*100);
disp(mean_CNMTE);

%% test trajectory plot
state_plot = 5;

test_fig = create_fig(44);
% add_label('sgtitle', 'Test Trajectories', 'font_size', 22);
test_fig.Position = [10 50 700 500];

axis_list = [];
j = 1;
for i = [1,3]
    set(gca, 'XTickLabel', []);
    ax_temp = add_subplot(2,1,j);
    axis_list = [axis_list, ax_temp];

    plot(yData{i,1}, yData{i,2}(state_plot,:), 'k', 'DisplayName', 'Test Trajectories', 'LineWidth', 3);
    plot(yRec{i,1}, yRec{i,2}(state_plot,:), 'g--', 'DisplayName', "SSM $\mathcal{O}$(5) Prediction", 'LineWidth', 3);
    add_label('y', "$\theta$");
    j = j + 1;
end

add_label('x', 'Time [s]', 'font_size', 25);
linkaxes(axis_list,'xy');

ylim([-20, 20]);
xlim([0,3]);


% legend
add_legend();
leg_handle = legend( 'Orientation', 'horizontal');


% Get current position of the legend
legendPos = get(leg_handle, 'Position');
% Calculate the center of the entire figure
figurePos = get(gcf, 'Position');
centerX = figurePos(3) / 2;
centerY = figurePos(4) / 2;
% Set new position for the legend to center it
newLegendPos = [(0.515 - legendPos(3)/2), (0.54 - legendPos(4)/2), legendPos(3:4)];
set(leg_handle, 'Position', newLegendPos);


% save figure
% save_fig(test_fig, 'shimmy_test_traj')

%% check backbone
traj_idx = 1;
BCC_state = 5;

% backbone curve info
zData = transformTrajectories(RDInfo.inverseTransformation.map, etaData);

rhoCal = max(abs(zData{traj_idx, 2}(1, :))) * 1.05;
amplitudeFunction = @(x) x(BCC_state,:);
BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, rhoCal);

% plot
shimmy_bbc = create_fig(11);
shimmy_bbc.Position = [10 50 700 700];
plot(BBCInfo.damping(1:940), BBCInfo.amplitude(1:940), 'DisplayName', "SSM $\mathcal{O}$(" + num2str(ROMOrder) + ") trained" +newline+ "on video data", 'LineWidth', 5);

% plot encoder model
load("./shimmy_data/encoder_bbc.mat");
plot(BBCInfo.damping(1:910), BBCInfo.amplitude(1:910), '--','DisplayName', "SSM $\mathcal{O}$(" + num2str(ROMOrder) + ") trained" +newline+ "on encoder data", 'LineWidth', 5);

xline(0, 'k-', 'DisplayName', '$x=0$', 'LineWidth', 5);


add_legend('font_size', 30);
legend('Location', 'northwest');
add_label('y', 'Amplitude [$\theta$]', 'font_size', 40);
add_label('x', 'Damping [1/s]', 'font_size', 40);
xlim([-1, 0.5]);


% save fig
% save_fig(shimmy_bbc, 'shimmy_damping_compare');

%% phase portrait
portrait_fig = create_fig(4);
portrait_fig.Position = [10 100 1000 700];
axis_tidy();
box off;
add_label('x', '$\theta(t)$', 'font_size', 40);
add_label('y', '$\theta(t+4\Delta t)$', 'font_size', 40);
% add_legend();

M = RDInfo.reducedDynamics.map; % dynamics in reduced coordinates


% forward time
time_to_integrate = linspace(0, 5, 5000);
y_0_forward = [19, 23;...
               0, 0];
for i = 1:size(y_0_forward, 2)
    [t, x] = ode15s(@(t,y) M(y), time_to_integrate, y_0_forward(:,i), odeset('AbsTol', 1e-20));
    eta_Traj = {};
    eta_Traj{1,1} = t';
    eta_Traj{1,2} = x';
    y_Traj = liftTrajectories(IMInfo, eta_Traj);
    % original
    plot(y_Traj{1,2}(1,:), y_Traj{1,2}(5,:), 'k', 'LineWidth', 2, 'HandleVisibility','off');
    % initial condition
    scatter(y_Traj{1,2}(1,1), y_Traj{1,2}(5,1), 100, 'b', 'filled', 'HandleVisibility','off');

end
% label
scatter(100, 100, 100, 'b', 'filled', 'displayname', 'Initial Conditions');

% LCO 
% amplitudes
theta = linspace(0,360, 500);
lco_amp_unstable = zeros(2,length(theta));
lco_amp_stable = zeros(2,length(theta));
for i = 1:length(theta)
    % unstable limit cycle
    z1 = (cosd(theta(i)) + 1i*sind(theta(i))) * 0.325038506157867;
    lco_amp_unstable(:,i) = [z1; conj(z1)];
    % stable limit cycle
    z2 = (cosd(theta(i)) + 1i*sind(theta(i))) * 0.464131769826263;
    lco_amp_stable(:,i) = [z2; conj(z2)];
end
% convert to physical coord
lco_zData = cell(2,2);
lco_zData{1,1} = 0;
lco_zData{2,1} = 0;
lco_zData{1,2} = lco_amp_unstable;
lco_zData{2,2} = lco_amp_stable;
eta_LCO = transformTrajectories(RDInfo.transformation.map, lco_zData);
y_LCO = liftTrajectories(IMInfo, eta_LCO);

% original
% unstable LCO
plot(y_LCO{1,2}(1,:), y_LCO{1,2}(5,:), 'r', 'LineWidth', 6, 'DisplayName', 'Unstable Limit Cycle');
% stable LCO
plot(y_LCO{2,2}(1,:), y_LCO{2,2}(5,:), 'g', 'LineWidth', 6, 'DisplayName', 'Stable Limit Cycle');


add_legend('font_size', 22);
legend('Location', 'northoutside', 'Orientation','horizontal');
grid off;
xlim([-23, 23]);
ylim([-20, 20]);

% amplitude line
xline(10.2632, 'k--', 'LineWidth', 3, 'HandleVisibility', 'off');
xline(15.0782, 'k--', 'LineWidth', 3, 'HandleVisibility', 'off');
txt = "$\theta = 10.3^{\circ} \rightarrow$";
text(9.8, -16, txt, 'Interpreter', 'latex', 'FontSize',30, 'HorizontalAlignment','right');
txt = '$\leftarrow \theta = 15.1^{\circ}$';
text(15.5, -16, txt,  'Interpreter', 'latex', 'FontSize', 30);

% output fig
% save_fig(portrait_fig, 'shimmy_phase_portrait');

%% limit cycles amplitude in normal coord

damp = RDInfo.conjugateDynamics.damping;

fun = @(z) damp(z);
opts = optimset('fzero');
opts.TolX = 1e-30;
y = fzero(fun, 0.5, opts);
disp(y);
% amplitudes
% z1 = 0.325038506157867
% z2 = 0.464131769826263