# Pacotes que serao utilizados --------------------------------------------

install.packages("doParallel")
library(tidymodels)
library(tidyverse)
library(ISLR)
library(vip)
library(doParallel)
library(skimr)


# Utiliza os dados da análise de dados do tri anterior
dados <- read.csv("https://raw.githubusercontent.com/PADSONL03/Atividade-Pratica-1/main/sao-paulo-properties-april-2019.csv", sep = ';')

dados %>% 
  skim()

# Faça um filtro nos dados considerando apenas os dados de aluguel (Negotiation.Type == “rent”), 
# separe os dados em dois conjuntos (treinamento e teste) e avalie o erro de previsão para os 
# seguintes modelos:

dados <- dados %>% 
  filter(Negotiation.Type == 'rent')

dados %>% 
  skim()

glimpse(dados)

# treinamento x teste -----------------------------------------------------
set.seed(123)

split <- initial_split(dados, prop = 0.7) ## , strata = "Price"

treinamento <- training(split)
teste <- testing(split)

glimpse(treinamento)

# recipes - processamento -------------------------------------------------


receita <- recipe(Price ~ ., data = treinamento) %>%
  step_rm(Negotiation.Type) %>% # remove o Negotiation.Type pois foi filtrada apenas com 'rent'%>%
  step_zv(all_predictors()) %>% # remove demais colunas que só tenham um único valor
  step_mutate(
    Elevator = as.factor(ifelse(Elevator == 1, "Sim", "Não")),
    Furnished = as.factor(ifelse(Furnished == 1, "Sim", "Não")),
    Swimming.Pool = as.factor(ifelse(Swimming.Pool == 1, "Sim", "Não")),
    New = as.factor(ifelse(New == 1, "Sim", "Não"))
  ) %>% #transforma flags em nominais para posteriormente serem transformadas em dummys
  step_integer(c(Latitude,Longitude)) %>% # transforma latitude e longitude em infos numéricas
  step_normalize(all_numeric(), -all_outcomes()) %>% #normaliza todas as colunas numéricas
  step_dummy(all_nominal(), -all_outcomes()) #transforma todas as categóricas em dummys


(receita_prep <- prep(receita)) # prepara a receita definida acima

treinamento_proc <- bake(receita_prep, new_data = NULL) # obtem os dados de treinamento processados

teste_proc <- bake(receita_prep, new_data = teste) # obtem os dados de teste processados

treinamento_proc %>% 
  glimpse()

treinamento_proc %>% 
  skim()

teste_proc %>% 
  skim()


# parsnip - modelos -------------------------------------------------------

# floresta aleatoria 

# Define o modelo

rf <- rand_forest() %>% # define o modelo floresta aleatoria
  set_engine("ranger", # define o pacote que vai fazer o ajuste do modelo
             importance = "permutation") %>%  #
  set_mode("regression") # define que é um modelo de regressao

# Treina o modelo

rf_fit <- rf %>% 
  fit(Price ~ ., treinamento_proc) # ajuste do modelo definido acima
rf_fit


# importância das variaveis
vip(rf_fit)

# Faz predições com teste -- Testa o modelo
fitted <- rf_fit %>% 
  predict(new_data = teste_proc) %>% # realiza predicao para os dados de teste
  mutate(observado = teste_proc$Price, # mesma estrutura do fitted_lm
         modelo = "random forest")

fitted %>% 
  head()


# yardstick - avaliar desempenho ------------------------------------------

fitted %>% 
  group_by(modelo) %>% # agrupa pelo modelo ajustado
  metrics(truth = observado, estimate = .pred) # obtem as metricas de avaliacao dos modelos


# tune - ajuste de hiperparametros ----------------------------------------

rf2 <- rand_forest(mtry = tune(), # definicao da floresta aleatoria 
                   trees = tune(), # todos argumentos com tune() serao tunados a seguir  
                   min_n = tune()) %>% 
  set_engine("ranger") %>% # define qual função sera usada
  set_mode("regression") # define que e'  problema de regressao
rf2


# validação cruzada para ajuste de hiperparametros
set.seed(123)
cv_split <- vfold_cv(treinamento, v = 10)

registerDoParallel() # pararaleliza o processo

# para tunar os parametros
rf_grid <- tune_grid(rf2, # especificacao do modelo
                     receita, # a receita a ser aplicada a cada lote
                     resamples = cv_split, # os lotes da validacao cruzada
                     grid = 10, # quantas combinacoes de parametros vamos considerar
                     metrics = metric_set(rmse, mae)) 

autoplot(rf_grid) # plota os resultados

rf_grid %>% 
  collect_metrics() 

rf_grid %>% 
  select_best("rmse") # seleciona a melhor combinacao de hiperparametros

best <- rf_grid %>% 
  select_best("rmse") # salva o melhor modelo na variavel best


# finaliza modelo
rf_fit2 <- finalize_model(rf2, parameters = best) %>% # informa os valores de hiperparametros a serem considerados
  fit(Price ~ ., treinamento_proc) # executa o modelo com os valores de hiperparametros definidos acima

fitted_rf2 <- rf_fit2 %>% # faz previsao para os dados de teste
  predict(new_data = teste_proc) %>% 
  mutate(observado = teste_proc$Price, 
         modelo = "random forest - tune")

fitted <- fitted %>% # empilha as previsoes da floresta tunada
  bind_rows(fitted_rf2)

fitted %>% # obtem as metricas de todos os modelos ajustados
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) 


# Boosting ----------------------------------------

boosting <- boost_tree(#mtry = tune(), # definicao do boosting
                   #trees = tune(), # todos argumentos com tune() serao tunados a seguir  
                   #min_n = tune()
  ) %>% 
  set_engine("xgboost") %>% # define qual função sera usada
  set_mode("regression") # define que e'  problema de regressao
boosting

# Treina o modelo

boosting_fit <- boosting %>% 
  fit(Price ~ ., treinamento_proc) # ajuste do modelo definido acima
boosting_fit


# importância das variaveis
vip(boosting_fit)

# Faz predições com teste -- Testa o modelo
fitted_bosting <- boosting_fit %>% # faz previsao para os dados de teste
  predict(new_data = teste_proc) %>% 
  mutate(observado = teste_proc$Price, 
         modelo = "boosting")

fitted <- fitted %>% # empilha as previsoes da floresta tunada
  bind_rows(fitted_bosting)


fitted %>% # obtem as metricas de todos os modelos ajustados
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) 



# tune - ajuste de hiperparametros ----------------------------------------

boosting_tuned <- boost_tree(mtry = tune(), # definicao do boosting
  trees = tune(), # todos argumentos com tune() serao tunados a seguir  
  min_n = tune(),
  tree_depth  = tune()
) %>% 
  set_engine("xgboost") %>% # define qual função sera usada
  set_mode("regression") # define que e'  problema de regressao
boosting_tuned



# validação cruzada para ajuste de hiperparametros
set.seed(123)
cv_split <- vfold_cv(treinamento, v = 10)

registerDoParallel() # pararaleliza o processo

# para tunar os parametros
boosting_grid <- tune_grid(boosting_tuned, # especificacao do modelo
                     receita, # a receita a ser aplicada a cada lote
                     resamples = cv_split, # os lotes da validacao cruzada
                     grid = 10, # quantas combinacoes de parametros vamos considerar
                     metrics = metric_set(rmse, mae)) 

autoplot(boosting_grid) # plota os resultados

boosting_grid %>% 
  collect_metrics() 

boosting_grid %>% 
  select_best("rmse") # seleciona a melhor combinacao de hiperparametros

best <- boosting_grid %>% 
  select_best("rmse") # salva o melhor modelo na variavel best


# finaliza modelo
boosting_fit2 <- finalize_model(boosting_tuned, parameters = best) %>% # informa os valores de hiperparametros a serem considerados
  fit(Price ~ ., treinamento_proc) # executa o modelo com os valores de hiperparametros definidos acima

fitted_boosting_tuned <- boosting_fit2 %>% # faz previsao para os dados de teste
  predict(new_data = teste_proc) %>% 
  mutate(observado = teste_proc$Price, 
         modelo = "boosting - tune")

fitted <- fitted %>% # empilha as previsoes da floresta tunada
  bind_rows(fitted_boosting_tuned)

fitted %>% # obtem as metricas de todos os modelos ajustados
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) 

# grafico de dispersao entre valor observado e valor predito
fitted %>% 
  ggplot(aes(observado, .pred, group = modelo, color = modelo)) + #eixo x observado, eixo y predito 
  geom_point(size = 2, alpha = .5) +  #, col = "blue"
  labs(y = "Predito", x = "Observado") +
  scale_y_log10(labels = scales::comma) +
  scale_x_log10(labels = scales::comma) +
  theme_minimal()+
  theme_bw()+
  facet_grid(~ modelo, scale = "free_y") 


