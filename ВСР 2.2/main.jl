# Подключение необходимых библиотек
using CSV
using DataFrames
using GLM
using Statistics
using Random
using Plots

# Загрузка данных
data = CSV.read("sales.csv", DataFrame)

# Предобработка данных
# Заполнение пропусков средними значениями
for col in names(data)
    if eltype(data[!, col]) == Float64
        data[!, col] = coalesce.(data[!, col], mean(data[!, col], skipmissing=true))
    end
end

# Кодирование категориальных переменных (если необходимо)
data = select(data, :item_id, :store_id, :date, :sales)  # выбираем нужные колонки

# Разделение данных на обучающую и тестовую выборки вручную
Random.seed!(42) # Устанавливаем семя для воспроизводимости
shuffle_indices = shuffle(1:size(data, 1))
train_size = Int(0.8 * size(data, 1)) # 80% для обучения
train_indices = shuffle_indices[1:train_size]
test_indices = shuffle_indices[train_size+1:end]

train_data = data[train_indices, :]
test_data = data[test_indices, :]

# Определение признаков и целевой переменной
X_train = train_data[:, Not(:sales)]
y_train = train_data[:, :sales]
X_test = test_data[:, Not(:sales)]
y_test = test_data[:, :sales]

# Построение модели линейной регрессии
model = lm(@formula(sales ~ item_id + store_id + date), train_data)

# Предсказание
y_pred = GLM.predict(model, test_data)

# Оценка модели
mse = mean((y_pred .- y_test).^2)
println("Среднеквадратичная ошибка: $mse")

# Коэффициенты модели
println("Коэффициенты модели:")
println(coef(model))

# Статистическая информация о модели
println("Статистическая информация:")
println(GLM.coeftable(model))

# Визуализация фактических и предсказанных значений
scatter(y_test, y_pred, label="Предсказанные значения", xlabel="Фактические значения", ylabel="Предсказанные значения", title="Сравнение фактических и предсказанных значений")
plot!([minimum(y_test), maximum(y_test)], [minimum(y_test), maximum(y_test)], label="Линия равенства", line=:dash)

# Визуализация распределения ошибок
errors = y_pred .- y_test
histogram(errors, bins=30, xlabel="Ошибка", ylabel="Частота", title="Распределение ошибок модели", legend=false)

# Вычисление R-квадрат
ss_res = sum((y_pred .- y_test).^2)  # Остаточная сумма квадратов
ss_tot = sum((y_test .- mean(y_test)).^2)  # Общая сумма квадратов
r_squared = 1 - (ss_res / ss_tot)
println("R-квадрат: $r_squared")

# Отображение графиков
plot!()  # Этот вызов завершает построение графиков