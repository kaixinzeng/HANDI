clear;
clc;
close('all');

%% load tracked data and encoder data
load('shimmy_data/measure1_encoder.mat');
shimmy_data = readtable('shimmy_data/measure1.csv');

%% encoder side by side
f1 = create_fig(1);

% encoder
plot(time, datadeg+0.75, 'k', 'DisplayName', 'Encoder', 'LineWidth', 1);
% tm
shift = 382;
plot(shimmy_data.time - shimmy_data.time(shift), shimmy_data.theta+0.4, 'b.--', 'DisplayName', 'Template Match', 'LineWidth', 1);

% setting
xlim([0, 300]);

%% tm data
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

%% encoder data new
xData_ref = {};

% lco
lco_start = 4718;
lco_end = lco_start + 200;
time_list = zeros(1,201);
angle_list = zeros(1,201);
for i = lco_start:lco_end
    [~, idx] = min(abs(time - (shimmy_data.time(i)-shimmy_data.time(shift))));
    time_list(i-lco_start+1) = time(idx);
    angle_list(i-lco_start+1) = datadeg(idx) + 0.75;
end
xData_ref{1,1} = time_list - time_list(1);
xData_ref{1,2} = angle_list;


lco_start = 6656;
lco_end = lco_start + 200;
time_list = zeros(1,201);
angle_list = zeros(1,201);
for i = lco_start:lco_end
    [~, idx] = min(abs(time - (shimmy_data.time(i)-shimmy_data.time(shift))));
    time_list(i-lco_start+1) = time(idx);
    angle_list(i-lco_start+1) = datadeg(idx) + 0.75;
end
xData_ref{2,1} = time_list - time_list(1);
xData_ref{2,2} = angle_list;

% decay
decay_start = 5353;
decay_end = 5499;
time_list = zeros(1,147);
angle_list = zeros(1,147);
for i = decay_start:decay_end
    [~, idx] = min(abs(time - (shimmy_data.time(i)-shimmy_data.time(shift))));
    time_list(i-decay_start+1) = time(idx);
    angle_list(i-decay_start+1) = datadeg(idx) + 0.75;
end
xData_ref{3,1} = time_list - time_list(1);
xData_ref{3,2} = angle_list;
xData_ref{3,2} = xData_ref{3,2} - mean(xData_ref{3,2});

decay_start = 2601;
decay_end = 2728;
time_list = zeros(1,128);
angle_list = zeros(1,128);
for i = decay_start:decay_end
    [~, idx] = min(abs(time - (shimmy_data.time(i)-shimmy_data.time(shift))));
    time_list(i-decay_start+1) = time(idx);
    angle_list(i-decay_start+1) = datadeg(idx) + 0.3;
end
xData_ref{4,1} = time_list - time_list(1);
xData_ref{4,2} = angle_list;
% xData_ref{4,2} = xData_ref{4,2} - mean(xData_ref{4,2});
%% plotting
% lco
lco_fig = create_fig(1);
lco_fig.Position = [10 100 600 400];
ylim([-20,20]);
plot_trajectory_id = 2;

plot(xData_ref{plot_trajectory_id,1}, xData_ref{plot_trajectory_id,2}, 'k', LineWidth=2);
plot(xData{plot_trajectory_id,1}, xData{plot_trajectory_id,2}, 'g.', MarkerSize=15);
add_label('y', "$\theta$");
add_label('x', 'Time [s]', 'font_size', 25);

% decay
decay_fig = create_fig(2);
decay_fig.Position = [700 100 600 400];
ylim([-20,20]);
plot_trajectory_id = 4;

plot(xData_ref{plot_trajectory_id,1}, xData_ref{plot_trajectory_id,2}, 'k', 'DisplayName', 'Encoder Reading', LineWidth=2);
plot(xData{plot_trajectory_id,1}, xData{plot_trajectory_id,2}, 'g.', 'DisplayName', 'Video-Extracted Data', MarkerSize=15);

add_legend();
add_label('y', "$\theta$");
add_label('x', 'Time [s]', 'font_size', 25);

% save figure
% save_fig(lco_fig, 'shimmy_error_1');
% save_fig(decay_fig, 'shimmy_error_2');

%%
% compute error
states_for_error = [1];
fullTrajDist = computeCNMTE(xData_ref, xData, states_for_error); % 2nd argument needs to be original data
fullTrajDist





