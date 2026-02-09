clear;
clc;
close all;

%% 1. 参数设置
addpath('npy-matlab');

% --- 定义要处理的8条轨迹的编号 ---
trajectory_indices_to_process = [1, 2, 3, 4, 5, 6, 7, 8];

% --- 为所有8条轨迹定义不同的起始裁剪时间（秒）---
% 这些值用于去掉每条轨迹开头的瞬态部分
start_offsets_seconds = [
    5.0, ... % Trajectory 1
    3.0, ... % Trajectory 2
    5.0, ... % Trajectory 3
    3.0, ... % Trajectory 4
    8.0, ... % Trajectory 5
    5.0, ... % Trajectory 6
    5.0, ... % Trajectory 7
    10.0 ... % Trajectory 8
];

% 视频帧率 (30 FPS)
fps = 30;
dt = 1/fps; % 计算每个时间步的真实时长

% 输出文件名
output_filename_npy = 'processed_trajectories_4D_8traj.npy';
output_filename_mat = 'processed_trajectories_4D_8traj.mat';

% --- 检查 writeNPY.m 是否存在 ---
if ~exist('writeNPY.m', 'file')
    error(['writeNPY.m not found. ' ...
        'Please download the complete "npy-matlab" toolbox from GitHub ' ...
        'and add it to your MATLAB path.']);
end

%% 2. 加载、计算速度并裁剪所有8条轨迹

disp('Loading preprocessed_data.mat...');
load("preprocessed_data.mat");

num_trajectories_to_process = length(trajectory_indices_to_process);
cropped_trajectories = cell(num_trajectories_to_process, 1);
cropped_lengths = zeros(num_trajectories_to_process, 1);

disp('Calculating velocities and cropping all 8 trajectories...');

for i = 1:num_trajectories_to_process
    % 获取真实的轨迹编号
    traj_idx = trajectory_indices_to_process(i);
    
    % 提取x和y坐标
    x = train_data(traj_idx).x;
    y = train_data(traj_idx).y;
    
    % 计算速度
    dx = diff(x) / dt;
    dy = diff(y) / dt;
    
    % 统一数据长度
    x_trimmed = x(1:end-1);
    y_trimmed = y(1:end-1);
    
    % 合并为4维状态向量 [x, dx, y, dy]
    trajectory_4D = [x_trimmed', dx', y_trimmed', dy'];
    
    % 从设置列表中获取对应的裁剪时间
    start_index_offset = round(start_offsets_seconds(traj_idx) * fps);
    
    fprintf('Processing Trajectory %d: Cropping first %.2f seconds (%d points).\n', ...
        traj_idx, start_offsets_seconds(traj_idx), start_index_offset);
    
    % 检查轨迹长度是否足够进行裁剪
    if size(trajectory_4D, 1) > start_index_offset
        cropped_trajectories{i} = trajectory_4D(start_index_offset + 1:end, :);
        cropped_lengths(i) = size(cropped_trajectories{i}, 1);
    else
        cropped_trajectories{i} = [];
        cropped_lengths(i) = 0;
        warning('Trajectory %d is too short for its custom crop. It will be ignored.', traj_idx);
    end
end

%% 3. 确定最短长度以统一所有轨迹

min_len = min(cropped_lengths(cropped_lengths > 0));
fprintf('\nAll 8 trajectories will be truncated to the minimum common length of %d points.\n', min_len);

%% 4. 构建最终的三维矩阵 (8 x T x 4)

final_data = zeros(num_trajectories_to_process, min_len, 4);
fprintf('Assembling the final %d x %d x 4 matrix...\n', num_trajectories_to_process, min_len);

for i = 1:num_trajectories_to_process
    if ~isempty(cropped_trajectories{i})
        temp_traj = cropped_trajectories{i};
        final_data(i, :, :) = temp_traj(1:min_len, :);
    end
end

%% 5. 保存为 .npy 和 .mat 文件

disp('-----------------------------------------');

writeNPY(final_data, output_filename_npy);
fprintf('Successfully saved data with shape (%d, %d, %d) to %s\n', ...
    size(final_data, 1), size(final_data, 2), size(final_data, 3), output_filename_npy);

% 保存更详细的 .mat 文件
saved_data.final_matrix = final_data;
saved_data.description = sprintf('Data for all 8 trajectories with custom start offsets.');
saved_data.original_indices_used = trajectory_indices_to_process;
saved_data.start_offsets_seconds = start_offsets_seconds;
saved_data.common_length_points = min_len;
saved_data.fps = fps;

save(output_filename_mat, '-struct', 'saved_data');
fprintf('Successfully saved detailed data structure to %s\n', output_filename_mat);

disp('-----------------------------------------');
fprintf('Done! Output shape: (8, %d, 4)\n', min_len);
fprintf('  - Dimension 1: 8 trajectories\n');
fprintf('  - Dimension 2: %d time steps\n', min_len);
fprintf('  - Dimension 3: 4 state variables [x, dx, y, dy]\n');
