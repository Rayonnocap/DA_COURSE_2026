import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 1-2. Импортируем данные, т.к предварительно посмотрели содержимое
# первую строку пропускаем, заголовок из 4 строк
# слова Missing и Offln тоже приняли за пропуски
df = pd.read_csv("data_example.csv", sep=",", skiprows=[0], header=[0, 1, 2, 3],
                 encoding="utf-8", encoding_errors="ignore",
                 na_values=["Missing", "Offln"])

# 3. Чистим заголовки
df.columns = df.columns.droplevel([1, 2, 3])
df.columns = [" ".join(c.replace("---", "").split()) for c in df.columns]
df = df.rename(columns={"Name": "time"})

# 4. Информация о датасете
print(df.shape)
df.info()

# 5. Время делаем индексом
df["time"] = pd.to_datetime(df["time"].str.strip(), format="%H:%M:%S %d/%m/%Y")
df = df.set_index("time")
df = df.sort_index()

# 6. Пропуски
print(df.isna().sum())
msno.matrix(df, sparkline=False)
plt.title("Пропуски в данных")
plt.show()

# 7. Оптимизация типов (оригинал df не меняем, работаем с копией df2)
df2 = df.copy()
for col in df2.columns:
    if df2[col].nunique() <= 10:
        df2[col] = df2[col].astype("category")   # мало разных значений - категория
    elif (df2[col].dropna() % 1 == 0).all():
        df2[col] = df2[col].astype("Int64")      # все значения целые (Int64 умеет хранить NaN поэтому вот так вот)

size1 = df.memory_usage(deep=True).sum() / 1024 ** 2
size2 = df2.memory_usage(deep=True).sum() / 1024 ** 2
print("Размер до оптимизации:", round(size1, 2), "МБ")
print("Размер после оптимизации:", round(size2, 2), "МБ")

# 8. Статистика Sair для устройств 21CT-30CT
cols_sair = [c for c in df.columns if "Sair" in c and 21 <= int(c[:2]) <= 30]
print(df[cols_sair].describe().T)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df[cols_sair])
plt.title("Boxplot температуры Sair, устройства 21CT-30CT")
plt.ylabel("Температура, °C")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 9. Ресемплирование по медиане за 4 минуты
df_res = df[cols_sair].resample("4min").median()
df_res.plot(figsize=(16, 6))
plt.title("Температура Sair, ресемплирование по медиане (4 мин)")
plt.xlabel("Время")
plt.ylabel("Температура, °C")
plt.legend(fontsize=7)
plt.show()

# 10. Сглаживание (скользящее среднее) для трёх признаков
df_smooth = df[cols_sair[:3]].rolling(200).mean()
df_smooth.plot(figsize=(16, 6))
plt.title("Сглаженная температура Sair (скользящее среднее, окно 200)")
plt.xlabel("Время")
plt.ylabel("Температура, °C")
plt.legend(fontsize=7)
plt.show()

# 11. Оригинал, ресемплирование и сглаживание на одном графике (первые 3 дня)
col = cols_sair[0]
original = df[col]["2021-03-01":"2021-03-03"]
resampled = df[col].resample("30min").median()["2021-03-01":"2021-03-03"]
smoothed = df[col].rolling(50).mean()["2021-03-01":"2021-03-03"]

plt.figure(figsize=(16, 6))
plt.plot(original, label="Оригинал", alpha=0.5)
plt.plot(resampled, label="Ресемплирование (30 мин)")
plt.plot(smoothed, label="Сглаживание (окно 50)")
plt.title("Сравнение обработки данных: " + col)
plt.xlabel("Время")
plt.ylabel("Температура, °C")
plt.legend()
plt.show()

# 12. Сколько устройств включено одновременно (состояние 0 = включено)
cols_state = [c for c in df.columns if "EKC состояние" in c]
count_on = (df[cols_state] == 0).sum(axis=1)
count_on.value_counts().sort_index().plot(kind="bar", figsize=(10, 5))
plt.title("Количество одновременно включённых устройств")
plt.xlabel("Сколько устройств включено")
plt.ylabel("Количество замеров")
plt.show()

# 13. Распределение по состояниям для трёх устройств
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    df[cols_state[i]].value_counts().sort_index().plot(kind="bar")
    plt.title(cols_state[i])
    plt.xlabel("Состояние")
    plt.ylabel("Количество замеров")
plt.tight_layout()
plt.show()

# 14. Матрица корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(df[cols_sair].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляция температуры Sair")
plt.tight_layout()
plt.show()

# 15. Гипотеза: когда устройство включено (состояние 0), температура u09 S5 ниже,
# чем когда оно в других состояниях
col_state = cols_state[0]
col_temp = col_state.replace("EKC состояние", "u09 S5 Темп")

mean_on = df.loc[df[col_state] == 0, col_temp].mean()
mean_off = df.loc[df[col_state] > 0, col_temp].mean()

sns.boxplot(x=df[col_state], y=df[col_temp])
plt.title("Температура u09 S5 при разных состояниях устройства")
plt.xlabel("Состояние")
plt.ylabel("Температура")
plt.show()

if mean_on < mean_off:
    print(f"Гипотеза подтвердилась, потому что при состоянии 0 средняя температура {mean_on:.2f}, "
          f"а в других состояниях {mean_off:.2f}. Мы предположили, что у включённого устройства температура ниже.")
else:
    print(f"Гипотеза не подтвердилась, потому что при состоянии 0 средняя температура {mean_on:.2f}, "
          f"а в других состояниях {mean_off:.2f}. Мы предположили, что у включённого устройства температура ниже.")
