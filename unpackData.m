function data = unpackData(filename)
data=struct;
lines = readlines(filename);
flagged_lines=[];
for i=1:length(lines)
    if(startsWith(lines{i}, "Pass"))
        flagged_lines(end+1) = i;
    end
end

data.N_pass = length(flagged_lines);
data.X = cell(1,data.N_pass);data.Y = cell(1,data.N_pass);
data.U = cell(1,data.N_pass);data.V = cell(1,data.N_pass);
for n=1:data.N_pass
    N_rows=0;
    N_cols=0;
    dataStart = 0;
    for i=flagged_lines(n):length(lines)
        if(startsWith(lines{i}, "Rows"))
            elems = split(lines{i}," ");
            N_rows = str2double(elems{end});
        end
        if(startsWith(lines{i}, "Cols"))
            elems = split(lines{i}," ");
            N_cols = str2double(elems{end});
        end
        if(startsWith(lines{i}, "image_x,image_y,U,V"))
            dataStart=i+1;
            break
        end
    end
    X = zeros(N_rows, N_cols);Y = zeros(N_rows, N_cols);
    U = zeros(N_rows, N_cols);V = zeros(N_rows, N_cols);
    for i=1:N_rows
        for j=1:N_cols
            line = lines{dataStart + (i-1)*N_cols + (j-1)};
            elems=split(line, ",");
            X(i,j) = str2double(elems{1});
            Y(i,j) = str2double(elems{2});
            U(i,j) = str2double(elems{3});
            V(i,j) = str2double(elems{4});
        end
    end
    data.X{n}=X;data.Y{n}=Y;
    data.U{n}=U;data.V{n}=V;
end


% vector validation
% for pass=1:data.N_pass
%     N_rows = size(data.X{pass},1);
%     N_cols = size(data.X{pass},2);
% 
%     threshold = 1;
% 
%     field_list = {"U", "V"};
%     for k=1:length(field_list)
%         field = field_list{k};
%         comparison_grid = zeros(size(data.X{pass}));
%         for i=1:N_rows
%             for j=1:N_cols%list of doubles
%                 neighbours = [];
%                 % search surroundings
%                 for ii=-1:1
%                     for jj=-1:1
%                         i_=i+ii;j_=j+jj;
%                         if(i_ >=1 && i_<=N_rows && j_>=1 && j_ <= N_cols)
%                             neighbours(end+1)=data.(field){pass}(i_,j_);
%                         end
%                     end
%                 end
%                 neighbours = mean(neighbours);
%                 comparison_grid(i,j) = abs((data.(field){pass}(i,j)-neighbours)/neighbours);
%             end
%         end
%         A=data.(field){pass};
%         X=data.X{pass};Y=data.Y{pass};
%         rng = comparison_grid(:)<threshold;
%         X_temp = X(rng);Y_temp = Y(rng);
%         A_temp = A(rng);
%         interpolater = scatteredInterpolant(X_temp,Y_temp,A_temp, 'natural');
%         rng = comparison_grid(:)>=threshold;
%         % disp(size(A));
%         % disp(size(X));
%         % disp(size(Y));
%         disp(sum(rng));
%         A(rng) = interpolater(X(rng),Y(rng));
%         data.(field){pass} = A;
%     end
% end

end

