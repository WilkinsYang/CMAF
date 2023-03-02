function signal=sonar_emulator(input_table,input_data,data_size,SP)
count=1;
invalid=0; %recording invalid data (i.e., the target too close to sonar)
result=zeros(data_size,301);
%start emulate the signal of each fish
while count<=data_size
    num=input_data(count,5);
    locx=zeros(num,1);
    locy=zeros(num,1);
    TS=zeros(num,1)+normrnd(0,2);
    file(count,1)=input_table{count+invalid,7};
    img(count,1:4)=input_table{count+invalid,8:11};
    for k=1:num
        locx(k)=input_data(count+invalid,3)/100;
        locy(k)=input_data(count+invalid,4)/100;
        TS(k)=input_data(count+invalid,2);
        [isopath{k}, channel{k},tgt{k},tgtplat{k}]=channel_model(SP,TS(k),locx(k), locy(k));
        
        wav = phased.RectangularWaveform(...
            'PulseWidth',SP.pulse_width,...
            'PRF',SP.prf,...
            'SampleRate',SP.fs);
        channel{k}.SampleRate = SP.fs;
    end
    if(locy(1)<1)
        invalid=invalid+1;
    else
           %% transmitter and receiver
        plat = phased.Platform(...
        'InitialPosition',SP.initial_position,...
        'Velocity',SP.velocity);

        proj = phased.IsotropicProjector(...
        'FrequencyRange',[0 SP.fc],'VoltageResponse',SP.VoltageResponse,'BackBaffled',true);

        [ElementPosition,ElementNormal] = helperSphericalProjector(8,SP.fc,SP.propSpeed);

        projArray = phased.ConformalArray(...
        'ElementPosition',ElementPosition,...
        'ElementNormal',ElementNormal,'Element',proj);

        hydro = phased.IsotropicHydrophone(...
        'FrequencyRange',[0 SP.fc],'VoltageSensitivity',SP.voltage_sensitivity);

        rx = phased.ReceiverPreamp(...
        'Gain',SP.gain,...
        'NoiseFigure',SP.noisefigure,...
        'SampleRate',SP.fs,...
        'SeedSource','Property',...
        'Seed',SP.seed);

           %% radiator and collector
        radiator = phased.Radiator('Sensor',projArray,'OperatingFrequency',...
        SP.fc,'PropagationSpeed',SP.propSpeed);

        collector = phased.Collector('Sensor',hydro,'OperatingFrequency',SP.fc,...
        'PropagationSpeed',SP.propSpeed);

            %% start emulation
         x = wav();    % Generate pulse    
        xmits = 1;
        rx_pulses1 = zeros(size(x,1),xmits);

        t = (0:size(x,1)-1)/SP.fs;
        for j = 1:xmits

            % Update target and sonar position
            [sonar_pos,sonar_vel] = plat(1/SP.prf);  

           for i = 1:num %Loop over targets

               [tgt_pos,tgt_vel] = tgtplat{i}(1/SP.prf);  

              % Compute transmission paths using the method of images. Paths are
              % updated according to the CoherenceTime property.
              [paths,dop,aloss,tgtAng,srcAng] = isopath{i}(...
                    sonar_pos,tgt_pos,...
                    sonar_vel,tgt_vel,1/SP.prf);  

              tsig = radiator(x,srcAng);

              % Propagate radiated signals through the channel
              tsig = channel{i}(tsig,paths,dop,aloss);

              % Target
              tsig = tgt{i}(tsig,tgtAng);

              % Collector
              rsig = collector(tsig,srcAng);
                
              % intrgrated signals
              rx_pulses1(:,j) = rx_pulses1(:,j)+rx(rsig);  

            end
        end
        rx_pulses1 = pulsint(rx_pulses1,'noncoherent');
        rx_pulse=rx_pulses1';
        result(count,:)=[0,rx_pulse];
      
        count=count+1;
    end
    disp(['count:',num2str(count+invalid), '. invalid:', num2str(invalid)]);
end
filename=string(file);
signal=table(filename,img,result);
disp(['total image number:',num2str(invalid),'. valid image number:',num2str(count)]);
end
