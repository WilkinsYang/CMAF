clf;
clc;
close all;
clear all;

%% parameter settings
smbass_total_image=1;
carp_total_image=1;
walleye_total_image=1;
sucker_total_image=1;
lamprey_total_image=1;
pike_total_image=1;
lmbass_total_image=1;

% path setting
PATH='your_path';   %modify your path here !

%channel parameter
SP.numPaths = 1;
SP.propSpeed = 1520;
SP.channelDepth = 30;
SP.fc=100e3;
SP.VoltageResponse=80;
SP.initial_position=[4.7083; 0; -15];
SP.velocity=[0; 0; 0];
SP.voltage_sensitivity=-140;
SP.maxRange = 15;                           % Maximum unambiguous range
SP.rangeRes = 0.1;                          % Required range resolution
SP.prf = SP.propSpeed/(2*SP.maxRange);            % Pulse repetition frequency
SP.pulse_width = 2*SP.rangeRes/SP.propSpeed;      % Pulse width
SP.pulse_bw = 1/SP.pulse_width;                % Pulse bandwidth
SP.fs = 2*SP.pulse_bw;                         % Sampling rate
SP.gain=40;
SP.noisefigure=10;
SP.seed=2007;

% data size
ds.smbass=3584;
ds.carp=2227;
ds.walleye=2306;
ds.sucker=16536;
ds.lamprey=3536;
ds.pike=381;
ds.lmbass=203;

%% read data
filename1=strcat(PATH,'\DIDSON_dataset\smbass_result_1fish.csv');
filename2=strcat(PATH,'\DIDSON_dataset\carp_result_1fish.csv');
filename3=strcat(PATH,'\DIDSON_dataset\walleye_result_1fish.csv');
filename4=strcat(PATH,'\DIDSON_dataset\sucker_result_1fish.csv');
filename5=strcat(PATH,'\DIDSON_dataset\lamprey_result_1fish.csv');
filename6=strcat(PATH,'\didson_custom_extra\pike_result_1fish.csv');
filename7=strcat(PATH,'\didson_custom_extra\lmbass_result_1fish.csv');

smbass_data=readtable(filename1);
smbass_input=table2array(smbass_data(:,2:6));

carp_data=readtable(filename2);
carp_input=table2array(carp_data(:,2:6));

walleye_data=readtable(filename3);
walleye_input=table2array(walleye_data(:,2:6));

sucker_data=readtable(filename4);
sucker_input=table2array(sucker_data(:,2:6));

lamprey_data=readtable(filename5);
lamprey_input=table2array(lamprey_data(:,2:6));

pike_data=readtable(filename6);
pike_input=table2array(pike_data(:,2:6));

lmbass_data=readtable(filename7);
lmbass_input=table2array(lmbass_data(:,2:6));

%% emulation
smbass_output=sonar_emulator(smbass_data, smbass_input, ds.smbass, SP);
disp('finish smbass emulation!');
carp_output=sonar_emulator(carp_data, carp_input, ds.carp, SP);
disp('finish carp emulation!');
walleye_output=sonar_emulator(walleye_data, walleye_input, ds.walleye, SP);
disp('finish walleye emulation!');
sucker_output=sonar_emulator(sucker_data, sucker_input, ds.sucker, SP);
disp('finish sucker emulation!');
lamprey_output=sonar_emulator(lamprey_data, lamprey_input, ds.lamprey, SP);
disp('finish lamprey emulation!');
pike_output=sonar_emulator(pike_data, pike_input, ds.pike, SP);
disp('finish pike emulation!');
lmbass_output=sonar_emulator(lmbass_data, lmbass_input, ds.lmbass, SP);
disp('finish lmbass emulation!');

%% output file
writetable(smbass_output,strcat(PATH,'\smbass_output_1fish.csv'));
writetable(carp_output,strcat(PATH,'\carp_output_1fish.csv'));
writetable(walleye_output,strcat(PATH,'\walleye_output_1fish.csv'));
writetable(sucker_output,strcat(PATH,'\sucker_output_1fish.csv'));
writetable(lamprey_output,strcat(PATH,'\lamprey_output_1fish.csv'));
writetable(pike_output,strcat(PATH,'\pike_output_1fish.csv'));
writetable(lmbass_output,strcat(PATH,'\lmbass_output_1fish.csv'));

