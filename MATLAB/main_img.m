% [Project]        DMP Comparison - Video Generation
% [Author]         Moses C. Nah
% [Creation Date]  Monday, Oct. 23th, 2022
%
% [Emails]         Moses C. Nah   : mosesnah@mit.edu

%% (--) INITIALIZATION

clear; close all; clc; workspace;

% Add the Libraries of 
cd( fileparts( matlab.desktop.editor.getActiveFilename ) );     
addpath( 'MATLAB_Library/myUtils', 'MATLAB_Library/myGraphics' )
myFigureConfig( 'fontsize',  20, ...
               'LineWidth',  10, ...
           'AxesLineWidth', 1.5, ...     For Grid line, axes line width etc.
              'markerSize',  25    )  
             
% Setting color structure 'c' as global variable
global c                                                                   
c  = myColor(); 

%% (--) Figure 3: Goal-directed Discrete Movement in Joint-space
% Figure 3 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches
clear data*; clc; close all;

fs = 40;
lw1 = 6; lw2= 9;

% Folder name
dir_name = '../main/example1_joint_discrete/data/';

file_name_DMP = [ dir_name, 'DMP.mat' ];
file_name_EDA = [ dir_name, 'EDA_Kq150_Bq50.mat' ];

data_DMP = load( file_name_DMP );
data_EDA = load( file_name_EDA );

% ======================================================================== %
% Plot1: Dynamic Movement Primitives, joint 1
subplot( 2, 2, 1 )
hold on
plot( data_DMP.t_arr, data_DMP.q_arr( :, 1 ), 'linewidth', lw2,  'color', c.blue );
plot( data_DMP.t_des, data_DMP.q_des( 1, : ), 'color', 'k', 'linewidth', lw1, 'linestyle', '--' );
set( gca, 'xlim', [ 0, 2.0 ], 'ylim', [0, 1.2], 'fontsize', fs  )
title( 'Dynamic Movement Primitives', 'fontsize', fs  )

% ======================================================================== %
% Plot2: Dynamic Movement Primitives, joint 2
subplot( 2, 2, 3 )
hold on
plot( data_DMP.t_arr, data_DMP.q_arr( :, 2 ), 'linewidth', lw2,  'color', c.blue );
plot( data_DMP.t_des, data_DMP.q_des( 2, : ), 'color', 'k', 'linewidth', lw1, 'linestyle', '--' );
set( gca, 'xlim', [ 0, 2.0 ], 'ylim', [0, 1.2], 'fontsize', 40 )
xlabel( '$t$ (sec)', 'fontsize', 40 )

% ======================================================================== %
% Plot3: Elementary Dynamic Actions, joint 1
subplot( 2, 2, 2 )
hold on
plot( data_EDA.t_arr, data_EDA.q_arr( :, 1 ), 'linewidth', lw2,  'color', c.orange );
plot( data_EDA.t_arr, data_EDA.q0_arr( :, 1 ), 'color', 'k', 'linewidth', lw1, 'linestyle', '--' );
set( gca, 'xlim', [ 0, 2.0 ], 'ylim', [0, 1.2], 'fontsize', fs  )
title( 'Elementary Dynamic Actions', 'fontsize', fs  )


% ======================================================================== %
% Plot3: Elementary Dynamic Actions, joint 2
subplot( 2, 2, 4 )
hold on
plot( data_EDA.t_arr, data_EDA.q_arr( :, 2 ), 'linewidth', lw2,  'color', c.orange );
plot( data_EDA.t_arr, data_EDA.q0_arr( :, 2 ), 'color', 'k', 'linewidth', lw1, 'linestyle', '--' );
set( gca, 'xlim', [ 0, 2.0 ], 'ylim', [0, 1.2], 'fontsize', 40 )
xlabel( '$t$ (sec)', 'fontsize', 40 )

mySaveFig( gcf, 'images/fig3' )

%% (--) Figure 4: Goal-directed Discrete Movement of EDA in Joint-space, sensitivity to Different Parameters
% Figure 4 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches

%% (--) ---- Figure 4a: Effect of Joint Stiffness

clc; close all; clear data*
fs = 40; lw2 = 5;
% Folder name
dir_name = '../main/example1_joint_discrete/data/';

% ======================================= %
% Figure 4a: Effect of Changing Stiffness
% ======================================= %
% Save the data as raw data
% The array of stiffness values used for the comparison
Kq = [ 30, 150, 1200 ];
Nk = length( Kq );
EDA_K = cell( 1, Nk );

for i = 1 : Nk
    EDA_K{ i } = load( [ dir_name, 'EDA_Kq', num2str( Kq(i) ), '_Bq50.mat' ] );
end

subplot( 2, 3, 1 )
hold on

t_arr = EDA_K{ 1 }.t_arr;
q_arr = EDA_K{ 1 }.q_arr( :, 1 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_K{ 1 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
ylabel( '$q_1(t)$ (rad)', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{K}_q=30\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 2 )
hold on
t_arr = EDA_K{ 2 }.t_arr;
q_arr = EDA_K{ 2 }.q_arr( :, 1 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange, 'linestyle', '-');
plot( t_arr, EDA_K{ 2 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
title( '$\mathbf{K}_q=150\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
set( gca, 'xticklabel', {} )

% ======================================================================== %
% Plot2: EDA, joint 2, Changing Stiffness
subplot( 2, 3, 3 )
hold on

t_arr = EDA_K{ 3 }.t_arr;
q_arr = EDA_K{ 3 }.q_arr( :, 1 );

plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-' );
plot( t_arr, EDA_K{ 1 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{K}_q=1200\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 4 )
hold on

t_arr = EDA_K{ 1 }.t_arr;
q_arr = EDA_K{ 1 }.q_arr( :, 2 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-');
plot( t_arr, EDA_K{ 1 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
ylabel( '$q_2(t)$ (rad)', 'fontsize', fs  )


subplot( 2, 3, 5 )
hold on

t_arr = EDA_K{ 2 }.t_arr;
q_arr = EDA_K{ 2 }.q_arr( :, 2 );   
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_K{ 2 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
xlabel( '$t$ (sec)', 'fontsize', 40 )

subplot( 2, 3, 6 )
hold on

t_arr = EDA_K{ 3 }.t_arr;
q_arr = EDA_K{ 3 }.q_arr( :, 2 );
    
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-' );
plot( t_arr, EDA_K{ 3 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )

mySaveFig( gcf, 'images/fig4a' )

%% (--) ---- Figure 4b: Effect of Joint Damping

clc; close all; clear data*
fs = 40; lw2 = 5;
% Folder name
dir_name = '../main/example1_joint_discrete/data/';

f = figure( );
Bq = [ 10, 50, 150];
Nb = length( Bq );
EDA_B = cell( 1, Nb );

for i = 1 : Nk
    EDA_B{ i } = load( [ dir_name, 'EDA_Kq150_Bq', num2str( Bq(i) ), '.mat' ] );
end

subplot( 2, 3, 1 )
hold on

t_arr = EDA_B{ 1 }.t_arr;
q_arr = EDA_B{ 1 }.q_arr( :, 1 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_B{ 1 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
ylabel( '$q_1(t)$ (rad)', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{B}_q=10\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 2 )
hold on
lw_s = { ':', '-', '-.' };  
lw2 = 5;
t_arr = EDA_B{ 2 }.t_arr;
q_arr = EDA_B{ 2 }.q_arr( :, 1 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange, 'linestyle', '-');
plot( t_arr, EDA_B{ 2 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
title( '$\mathbf{B}_q=50\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 3 )
hold on

t_arr = EDA_B{ 3 }.t_arr;
q_arr = EDA_B{ 3 }.q_arr( :, 1 );

plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-' );
plot( t_arr, EDA_B{ 1 }.q0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{B}_q=150\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 4 )
hold on

t_arr = EDA_B{ 1 }.t_arr;
q_arr = EDA_B{ 1 }.q_arr( :, 2 );
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-');
plot( t_arr, EDA_B{ 1 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
ylabel( '$q_2(t)$ (rad)', 'fontsize', fs  )


subplot( 2, 3, 5 )
hold on

t_arr = EDA_B{ 2 }.t_arr;
q_arr = EDA_B{ 2 }.q_arr( :, 2 );   
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_B{ 2 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
xlabel( '$t$ (sec)', 'fontsize', 40 )

subplot( 2, 3, 6 )
hold on

t_arr = EDA_B{ 3 }.t_arr;
q_arr = EDA_B{ 3 }.q_arr( :, 2 );
    
plot( t_arr, q_arr, 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_B{ 3 }.q0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )

mySaveFig( gcf, 'images/fig4b' )

%% (--) Figure 5: Goal-directed Discrete Movement in Task-space
% Figure 5, subfigures on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches

%% (--) ---- Figure 5a: Goal-directed, without Stretch

clc; close all; clear data*
fs = 40;

% Folder name
dir_name = '../main/example2_task_discrete/data/';

% =============================== Motor Primitives ==================================== %
data_DMP = load( [ dir_name, 'DMP.mat' ] );
data_EDA = load( [ dir_name, 'EDA_Kp60_Bp20.mat' ] );

% Saving the start and end positions
% Get the start and end of movements
g_start = data_EDA.p0_arr(    1, : );
g_end   = data_EDA.p0_arr( end, : );

% ======================================================================== %
% Plot1: Dynamic Movement Primitives
subplot( 1, 2, 1 )
hold on 

% Get the x, y position of the joints 
q_abs = cumsum( data_DMP.q_arr , 2 );
x_arr = cumsum( cos( q_abs ), 2 );
y_arr = cumsum( sin( q_abs ), 2 );

alpha_arr = [0.2, 0.3, 0.3, 1.0];
idx_arr   = [1, 400, 600, 2000];

for i = 1 : length( idx_arr )
    idx = idx_arr( i );
    alpha = alpha_arr( i );
    scatter( [ 0, x_arr( idx, 1:end-1 ) ], [ 0, y_arr( idx, 1:end-1 ) ], 400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'MarkerFaceAlpha', alpha,'MarkerEdgeAlpha',alpha  )
    p2 = plot( [ 0, x_arr( idx, : ) ], [ 0, y_arr( idx, : ) ], 'color', c.black, 'linewidth', 4 );
    p2.Color( 4 ) = alpha;
    scatter( x_arr( idx, end ), y_arr( idx, end),  1200,  'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'MarkerFaceAlpha', alpha,'MarkerEdgeAlpha',alpha  )
end

plot( x_arr( :, 2 ), y_arr( :, 2 ), 'linewidth', 8, 'color', c.blue )
plot( data_DMP.p_des( 1, : ), data_DMP.p_des( 2, : ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

% Start and End Location
scatter( 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0 )
scatter( 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0 )

text( -0.75, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( -0.75, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )

xlabel( '$X$ (m)', 'fontsize', fs );
ylabel( '$Y$ (m)', 'fontsize', fs )
axis equal

set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs ) 

% ======================================================================== %
% Plot2: Elementary Dynamic Actions
subplot( 1, 2, 2)
hold on

alpha_arr = [0.2, 0.3, 0.3, 1.0];
idx_arr   = [1, 400, 600, 2000];

% Get the x, y position of the joints 
q_abs = cumsum( data_EDA.q_arr , 2 );
x_arr = cumsum( cos( q_abs ), 2 );
y_arr = cumsum( sin( q_abs ), 2 );


for i = 1 : length( idx_arr )
    idx = idx_arr( i );
    alpha = alpha_arr( i );
    scatter( [ 0, x_arr( idx, 1:end-1 ) ], [ 0, y_arr( idx, 1:end-1 ) ], 400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'MarkerFaceAlpha', alpha,'MarkerEdgeAlpha',alpha  )
    p2 = plot( [ 0, x_arr( idx, : ) ], [ 0, y_arr( idx, : ) ], 'color', c.black, 'linewidth', 4 );
    p2.Color( 4 ) = alpha;
    scatter( x_arr( idx, end ), y_arr( idx, end),  1200,  'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'MarkerFaceAlpha', alpha,'MarkerEdgeAlpha',alpha  )

end
plot( data_EDA.p_arr(  :, 1 ) , data_EDA.p_arr( :, 2 ), 'linewidth', 8, 'color', c.orange )
plot( data_EDA.p0_arr( :, 1 ), data_EDA.p0_arr( :, 2 ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

% Start and End Location
scatter( 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0 )
scatter( 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0 )

text( -0.75, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( -0.75, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )

xlabel( '$X$ (m)', 'fontsize', fs )
axis equal
set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs ) 

mySaveFig( gcf, 'images/fig5a' )

%% (--) ---- Figure 5b: Goal-directed, Stretched


%% (--) Figure 6: Goal-directed Discrete Movement of EDA in Task-space, sensitivity to Different Parameters

%% (--) ---- Figure 6b: Effect of Translational Stiffness

clc; close all; clear data*
fs = 40; lw2 = 5;
% Folder name
dir_name = '../main/example2_task_discrete/data/';


Kp = [ 20, 60, 300 ];
Nk = length( Kp );
EDA_K = cell( 1, Nk );

for i = 1 : Nk
    EDA_K{ i } = load( [ dir_name, 'EDA_Kp', num2str( Kp(i) ), '_Bp20.mat' ] );
end

subplot( 2, 3, 1 )
hold on

t_arr = EDA_K{ 1 }.t_arr;
plot( t_arr, EDA_K{ 1 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_K{ 1 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
ylabel( '$X$ (m)', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
title( '$\mathbf{K}_p=20\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 2 )
hold on
t_arr = EDA_K{ 2 }.t_arr;
plot( t_arr, EDA_K{ 2 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange, 'linestyle', '-');
plot( t_arr, EDA_K{ 2 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
title( '$\mathbf{K}_p=60\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
set( gca, 'xticklabel', {} )

% ======================================================================== %
% Plot2: EDA, joint 2, Changing Stiffness
subplot( 2, 3, 3 )
hold on
t_arr = EDA_K{ 3 }.t_arr;
plot( t_arr, EDA_K{ 3 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-' );
plot( t_arr, EDA_K{ 3 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{K}_p=300\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
set( gca, 'xticklabel', {} )


subplot( 2, 3, 4 )
hold on

t_arr = EDA_K{ 1 }.t_arr;
plot( t_arr, EDA_K{ 1 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_K{ 1 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

subplot( 2, 3, 5 )
hold on

t_arr = EDA_K{ 2 }.t_arr;
plot( t_arr, EDA_K{ 2 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_K{ 2 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

subplot( 2, 3, 6 )
hold on

t_arr = EDA_K{ 3 }.t_arr;
plot( t_arr, EDA_K{ 3 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_K{ 3 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

mySaveFig( gcf, 'images/fig6a' )

%% (--) ---- Figure 6b: Effect of Translational Damping

clc; close all; clear data*
fs = 40; lw2 = 5;
% Folder name
dir_name = '../main/example2_task_discrete/data/';

f = figure( ); a = axes( 'parent', f );
Bq = [ 10, 20, 60];
Nb = length( Bq );
EDA_B = cell( 1, Nb );


for i = 1 : Nk
    EDA_B{ i } = load( [ dir_name, 'EDA_Kp60_Bp', num2str( Bq(i) ), '.mat' ] );
end

subplot( 2, 3, 1 )
hold on

t_arr = EDA_B{ 1 }.t_arr;
plot( t_arr, EDA_B{ 1 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_B{ 1 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
ylabel( '$X$ (m)', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
title( '$\mathbf{B}_p=10\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 2 )
hold on
t_arr = EDA_B{ 2 }.t_arr;
plot( t_arr, EDA_B{ 2 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange, 'linestyle', '-');
plot( t_arr, EDA_B{ 2 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
title( '$\mathbf{B}_p=20\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
set( gca, 'xticklabel', {} )

subplot( 2, 3, 3 )
hold on
t_arr = EDA_B{ 3 }.t_arr;
plot( t_arr, EDA_B{ 3 }.p_arr(  :, 1 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-' );
plot( t_arr, EDA_B{ 3 }.p0_arr( :, 1 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0, 1.5], 'fontsize', fs  )
title( '$\mathbf{B}_p=60\mathbf{I}_2$', 'fontsize', fs  )
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [-0.4, 0.4001], 'fontsize', fs  )
set( gca, 'xticklabel', {} )


subplot( 2, 3, 4 )
hold on

t_arr = EDA_B{ 1 }.t_arr;
plot( t_arr, EDA_B{ 1 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange );
plot( t_arr, EDA_B{ 1 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

subplot( 2, 3, 5 )
hold on

t_arr = EDA_B{ 2 }.t_arr;
plot( t_arr, EDA_B{ 2 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_B{ 2 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

subplot( 2, 3, 6 )
hold on

t_arr = EDA_B{ 3 }.t_arr;
plot( t_arr, EDA_B{ 3 }.p_arr(  :, 2 ), 'linewidth', lw2,  'color', c.orange,'linestyle', '-'  );
plot( t_arr, EDA_B{ 3 }.p0_arr( :, 2 ), 'linewidth', 6, 'color', 'k', 'linestyle', ':' );
set( gca, 'xlim', [ 0, 1.5 ], 'ylim', [0.5, 2], 'fontsize', fs  )
ylabel( '$Y$ (m)', 'fontsize', fs  )

mySaveFig( gcf, 'images/fig6b' )


%% (--) Figure 7

%% (--) Figure 8

