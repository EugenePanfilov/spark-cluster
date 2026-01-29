#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType
import subprocess

# ----------------------------
# НАСТРОЙКИ (меняете здесь)
# ----------------------------
SRC = "hdfs:///user/ubuntu/data/"            # вход: папка с txt/csv
DST = "hdfs:///user/ubuntu/parquet/"         # выход: папка parquet
SEP = ","                                    # разделитель CSV
TS_FORMAT = "yyyy-MM-dd HH:mm:ss"            # формат tx_datetime

PLOT_LOCAL = "/tmp/fraud_per_week.png"       # где сохранить PNG локально на драйвере
PLOT_HDFS_DIR = "hdfs:///user/ubuntu/plots/" # куда положить PNG в HDFS
PLOT_HDFS_PATH = PLOT_HDFS_DIR.rstrip("/") + "/fraud_per_week.png"

BAD_HEADER = "# tranaction_id | tx_datetime | customer_id | terminal_id | tx_amount | tx_time_seconds | tx_time_days | tx_fraud | tx_fraud_scenario"

# ----------------------------
# Spark
# ----------------------------
print("[Spark] Создаю SparkSession...", flush=True)
spark = SparkSession.builder.appName("FRAUD_PREPROCESS_SIMPLE").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("[Spark] SparkSession готов. Уровень логов Spark = ERROR.", flush=True)

# 1) Чтение сырых файлов: 9 колонок как строки
print("\n[1] Читаю сырые файлы из HDFS:", flush=True)
print(f"    SRC = {SRC}", flush=True)
raw_schema = StructType([StructField(f"_c{i}", StringType(), True) for i in range(9)])

df = (spark.read
      .option("header", "false")
      .option("sep", SEP)
      .option("mode", "PERMISSIVE")
      .schema(raw_schema)
      .csv(SRC))
print("[1] Чтение сырых файлов завершено (DataFrame создан).", flush=True)

# 2) Чистка мусора
print("\n[2] Чищу мусор: удаляю пустые строки и строку BAD_HEADER (если она встречается в данных)...", flush=True)
df = df.filter(F.col("_c0").isNotNull()).filter(F.col("_c0") != BAD_HEADER)
print("[2] Чистка завершена.", flush=True)

# 3) Переименование колонок
print("\n[3] Переименовываю колонки _c0.._c8 в осмысленные имена...", flush=True)
df = df.toDF(
    "transaction_id", "tx_datetime", "customer_id", "terminal_id", "tx_amount",
    "tx_time_seconds", "tx_time_days", "tx_fraud", "tx_fraud_scenario"
)
print("[3] Переименование завершено.", flush=True)

# 4) Касты типов + парсинг datetime
print("\n[4] Привожу типы колонок и парсю tx_datetime в timestamp:", flush=True)
print(f"    TS_FORMAT = {TS_FORMAT}", flush=True)
df = (df
      .withColumn("transaction_id", F.col("transaction_id").cast("long"))
      .withColumn("tx_datetime", F.to_timestamp("tx_datetime", TS_FORMAT))
      .withColumn("customer_id", F.col("customer_id").cast("long"))
      .withColumn("terminal_id", F.col("terminal_id").cast("long"))
      .withColumn("tx_amount", F.col("tx_amount").cast("double"))
      .withColumn("tx_time_seconds", F.col("tx_time_seconds").cast("long"))
      .withColumn("tx_time_days", F.col("tx_time_days").cast("int"))
      .withColumn("tx_fraud", F.col("tx_fraud").cast("int"))
      .withColumn("tx_fraud_scenario", F.col("tx_fraud_scenario").cast("int"))
)
print("[4] Приведение типов завершено.", flush=True)

# 5) Считаем NULL tx_datetime и удаляем их
print("\n[5] Считаю строки, где tx_datetime стало NULL после парсинга (будут удалены)...", flush=True)
n_null_dt = df.filter(F.col("tx_datetime").isNull()).count()
print("    Количество строк с NULL tx_datetime:", flush=True)
print(f"    {n_null_dt}", flush=True)

print("[5] Удаляю строки с NULL tx_datetime...", flush=True)
df = df.filter(F.col("tx_datetime").isNotNull())
print("[5] Удаление завершено.", flush=True)

# 6) Пишем Parquet
print("\n[6] Записываю датасет в Parquet (overwrite) в HDFS:", flush=True)
print(f"    DST = {DST}", flush=True)
df.write.mode("overwrite").parquet(DST)
print("[6] Запись Parquet завершена.", flush=True)

# 7) Читаем Parquet обратно и делаем контрольные выводы
print("\n[7] Читаю Parquet обратно из HDFS для контроля:", flush=True)
dfp = spark.read.parquet(DST)
print("[7] Чтение Parquet завершено.", flush=True)

print("\n[7] Схема датасета после записи/чтения Parquet:", flush=True)
dfp.printSchema()

print("[7] Первые 5 строк датасета:", flush=True)
dfp.show(5, truncate=False)

print("[7] Общее количество строк:", flush=True)
print(dfp.count(), flush=True)

# 8) Fraud по неделям
print("\n[8] Агрегация fraud-операций по неделям (week_start -> n_fraud)...", flush=True)
weekly = (dfp
  .withColumn("week_start", F.date_trunc("week", F.col("tx_datetime")).cast("date"))
  .groupBy("week_start")
  .agg(F.sum(F.col("tx_fraud")).alias("n_fraud"))
  .orderBy("week_start")
)
print("[8] Агрегация готова. Показываю первые 5 строк (неделя -> n_fraud):", flush=True)
weekly.show(5, truncate=False)

# 9) Рисуем график и сохраняем PNG локально
print("\n[9] Строю график и сохраняю PNG локально (на драйвере):", flush=True)
print(f"    PLOT_LOCAL = {PLOT_LOCAL}", flush=True)

import matplotlib
matplotlib.use("Agg")  # batch backend
import matplotlib.pyplot as plt

print("[9] Конвертирую weekly в pandas (обычно недель мало)...", flush=True)
pdf = weekly.toPandas()
print(f"[9] Количество недель в pandas: {len(pdf)}", flush=True)

plt.figure(figsize=(12, 4))
plt.bar(pdf["week_start"].astype(str), pdf["n_fraud"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("week_start")
plt.ylabel("n_fraud")
plt.tight_layout()
plt.savefig(PLOT_LOCAL, dpi=200, bbox_inches="tight")
plt.close()
print("[9] PNG сохранён локально.", flush=True)

# 10) Автозагрузка PNG в HDFS
print("\n[10] Загружаю PNG в HDFS:", flush=True)
print(f"     PLOT_HDFS_PATH = {PLOT_HDFS_PATH}", flush=True)

subprocess.run(["hdfs", "dfs", "-mkdir", "-p", PLOT_HDFS_DIR], check=True)
subprocess.run(["hdfs", "dfs", "-put", "-f", PLOT_LOCAL, PLOT_HDFS_PATH], check=True)

print("[10] Готово: PNG загружен в HDFS.", flush=True)

print("\n[Done] Останавливаю SparkSession.", flush=True)
spark.stop()
print("[Done] Завершено.", flush=True)