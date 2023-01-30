% DCC40 - Atividade 03
% Reconhecimento de padrões via RNA
% Abordagem RBF com algoritmo WTA

% Lista de módulos
% 1 - Leitor da base de dados
% 2 - Execução inicial


% 1 - Leitor da base de dados

linhasBaseOriginal = readlines("C:\Users\oficial\Documents\DCCMAPI\Atv03\car.data");
baseOriginal = -1 .* ones(length(linhasBaseOriginal)-1 ,7);

for i=1:length(linhasBaseOriginal)-1
    partesLinha = split(linhasBaseOriginal(i,1),",");
    for j=1:7
        if (partesLinha(j,1) == "low" || partesLinha(j,1) == "small" || partesLinha(j,1) == "unacc" || partesLinha(j,1) == "2")
            baseOriginal(i,j) = 0;
        elseif (partesLinha(j,1) == "med" || partesLinha(j,1) == "3" || partesLinha(j,1) == "acc")
            baseOriginal(i,j) = 1;
         elseif (partesLinha(j,1) == "high" || partesLinha(j,1) == "big" || partesLinha(j,1) == "good" || partesLinha(j,1) == "more")   
            baseOriginal(i,j) = 2; 
         elseif (partesLinha(j,1) == "vhigh" || partesLinha(j,1) == "5more" || partesLinha(j,1) == "vgood")   
            baseOriginal(i,j) = 3;
         elseif partesLinha(j,1) == "4"
             if j == 3 
                 baseOriginal(i,j) = 2;
             else 
                 baseOriginal(i,j) = 1;
             end
        end
    end
end



% 2 - Ciclo inicial de 10 treinamentos

% Inicializar entradas e neurônios

tamanhoBase = length(linhasBaseOriginal);
matrizEntrada = baseOriginal(:,1:6)';
matrizEntrada = normalize(matrizEntrada);
D = normalize(baseOriginal (:,7)'); % Valores corretos de saída

I = size(matrizEntrada,1);

O = 1;

epocas = 10;
treinamento = 1;


ValoresDeSaida = [];
OcorrenciasDeEQs = [];

while treinamento < 11

eta1 = 0.08; %taxa de aprendizagem
H = randi(12);%size(matrizEntrada,1);

inicioIteracao = randi(1727);
if inicioIteracao - epocas < 0 
    inicioIteracao = inicioIteracao + epocas;
else if inicioIteracao + epocas > tamanhoBase 
        inicioIteracao = inicioIteracao - epocas;
     end
end

iteracao = inicioIteracao;
% inicializa aleatoriamente os centros
centros = [(normalize(randperm(100,I),'range') - 0.5) .* (normalize(randperm(100,H),'range')' - 0.5)]; 
PesosHO = normalize(randperm(100,H),'range')' - 0.5;


while iteracao < (inicioIteracao+epocas)
    EqUltimo = 10000;
    Eq = 9999;
    AcumulaEq = 0;

   while Eq < EqUltimo % Eq está diminuindo ? 
    X = matrizEntrada(1:6,iteracao+1);
    
    % inicializa o mi -> miDaSig
    miDaSig = ones(size(X));
    % definicao dos centros
    
    for i=1:length(X)
      proximo = 10;
      for j=1:length(centros)
           if sqrt((X(i) - centros(i,j))^2) < proximo 
              proximo = sqrt((X(i) - centros(i,j))^2);
              linha = i;
              coluna = j;
              miDaSig(i) = proximo; 
          end
      end
      centros(linha,coluna) = centros(linha,coluna) + eta1 * (X(linha) - centros(linha,coluna));
      AcumulaEq = AcumulaEq + (X(linha) - centros(linha,coluna))^2;
    end
   EqUltimo = Eq;
   Eq = AcumulaEq / length(X);
   
end
   
   % Camada escondida
   % Sigma de raio único
   % Cálculo das aberturas
   Dmax = 0;
   for i=1:size(centros,1)
       for j=i:size(centros,2)
           for f=j+1:size(centros,1)
             Daux = sqrt(abs((centros(i,j)^2 - centros(i,f)^2)));
             if Dmax < Daux
               Dmax = Daux; 
             end
           end
       end    
   end

   % aplica gaussiana
   k = 0.7;
   sigma = k * Dmax;
   NHSaida = exp(-((miDaSig .^ 2) / 2 * (sigma .^ 2)));
   DerivadaSaida = NHSaida - (NHSaida .^ 2);
   Saida = NHSaida .* PesosHO;

   % Camada de saída
   eta2 = 0.06;
   Erro = Saida - D(iteracao+1); 

   ErroQ = mse(Erro);
   DeltaP = eta2 * ErroQ * DerivadaSaida .* Saida;
   PesosHO = PesosHO + DeltaP;
   OcorrenciasDeEQs(end+1)=ErroQ;
   iteracao = iteracao+1;
end
 
ValoresDeSaida(end+1) = mse(Saida);
treinamento = treinamento + 1;

end
figure (1), clf
hold on
plot(ValoresDeSaida, 'r');
title("Resultados RBF - Variando camada escondida ");
xlabel('Treinamento');
ylabel('Saídas');

figure (2), clf
hold on
plot(OcorrenciasDeEQs,'b');
title("Erro da rede - Variando camada escondida");
xlabel('Épocas');
ylabel('Erro encontrado');

