import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from PIL import Image

df = pd.read_csv('tutihasi_00001_20241001083012_SMPINF.csv', dtype=str, encoding='shift_jis')#csv読み込みエンコードをshift_jisに指定（日本語対応）

df_select = df.loc[:,["物標ID","時:分:秒.ミリ秒","速度(m/s)","方位角(度)", "緯度", "経度"]]#必要なカラムを抽出
df_select = df_select.rename(columns={"速度(m/s)": "speed", "物標ID": "target_id"})#エラーが出たので、カラムを別の表記にする！！！！！

#数字として用いたい列をfloat型に変換
df_select["speed"] = df_select["speed"].astype(float)
df_select["方位角(度)"] = df_select["方位角(度)"].astype(float)

#速度に外れ値が存在するので、まず外れ値を除去する（30m/s以下）
df_select = df_select.query("0 <= speed < 30")

#ID別、時間順にデータをソートし、インデックスをリセット
df_sorted = df_select.sort_values(["target_id","時:分:秒.ミリ秒"]).reset_index()

#とにかく加速度（dv/dt）を計算して、値の小さい順に並べて、順番に物標IDを10個入手する
df_sorted["time"] = pd.to_datetime(df_sorted["時:分:秒.ミリ秒"], format="%H:%M:%S.%f")

df_sorted["dt"] = df_sorted.groupby("target_id")["time"].diff()#物標IDごとに前の行との差分をとる
df_sorted["dt"] = df_sorted["dt"].dt.total_seconds()

#dtが0の行は計算できないので省く
df_sorted = df_sorted.query("dt != 0")
df_sorted["speed"] = df_sorted["speed"].astype(float)
df_sorted["dv"] = df_sorted.groupby("target_id")["speed"].diff().fillna(0.0)#物標IDごとに前の行との差分をとる
df_sorted["a"] = df_sorted["dv"].astype(float) / df_sorted["dt"].astype(float)#加速度を出す

#aを小さい順に並べる
df_sorted_a = df_sorted.sort_values("a")

#aの小さい方から10個リストに抜き出す
top10_id_lst = []
for id_num in df_sorted_a["target_id"]:
    if len(top10_id_lst) >= 10:
        break
    elif id_num not in top10_id_lst:
        top10_id_lst.append(id_num)

#入手したidに該当するデータを抜き出す
df_top10 = df_sorted[df_sorted["target_id"].isin(top10_id_lst)]
df_top10 = df_top10.sort_values(["target_id","時:分:秒.ミリ秒"]).reset_index()#ID別、時間順にデータをソートし、インデックスをリセット
df_top10["方位角(度)"] = np.radians(df_top10["方位角(度)"].astype(float))
df_top10["dv_x"] = df_top10["speed"] * np.sin(df_top10["方位角(度)"].astype(float))#地図の東西方向をｘ方向とする
df_top10["dv_y"] = df_top10["speed"] * np.cos(df_top10["方位角(度)"].astype(float))#地図の南北方向をＹ方向とする

df_top10["dx"] = df_top10["dv_x"] * df_top10["dt"]#ｘ方向の移動量を計算
df_top10["dy"] = df_top10["dv_y"] * df_top10["dt"]#Ｙ方向の移動量を計算

df_x0y0 = pd.read_csv('top10_x0y0data.csv', encoding='shift_jis')#地図上の初期位置を指定（これは別データで用意。緯度経度から座標を外部サイトで計算したため。）

i = 0
def update_first_dx(group):#グループの先頭のdxの値を初期値に書き換える関数を用意
    global i
    group.iloc[0] = df_x0y0["x0"][i]
    i += 1
    return group

j = 0
def update_first_dy(group):#グループの先頭のdyの値を初期値に書き換える関数を用意
    global j
    group.iloc[0] = df_x0y0["y0"][j]
    j += 1
    return group

df_top10["dx"] = list(df_top10.groupby("target_id")["dx"].apply(update_first_dx))#dxの値に関数を適用
df_top10["dy"] = list(df_top10.groupby("target_id")["dy"].apply(update_first_dy))#dyの値に関数を適用

df_top10["x"] = df_top10.groupby("target_id")["dx"].cumsum()#dxを累積してｘの値を出す
df_top10["y"] = df_top10.groupby("target_id")["dy"].cumsum()#dyを累積してｘの値を出す

df_top10["x"] = df_top10["x"].astype(float)
df_top10["y"] = df_top10["y"].astype(float)

df_top10.to_csv('top10_data.csv', encoding='shift_jis')#top10のデータを取り出した新しいCSVファイルを作成

#動きを可視化してアニメーションで表示

fig, ax = plt.subplots(figsize=(8,8))

groups = df_top10.groupby("target_id")#IDごとにグループ化

frames = []#アニメーションに使うグラフを保存するリストを準備

for i in range(0, groups.size().max()):#IDグループの中で一番多いデータ数まで繰り返し
    artists = []#アニメーションに使うグラフを保存するリストを準備、1フレームごとに初期化
    for id_num, group in groups:
        if i < len(group):#フレーム数がデータ最大数より小さければそこまでの線を保存
            x_data = group.loc[:i, "x"]
            y_data = group.loc[:i, "y"]
        else:#フレーム数がデータ最大数より大きければ最後の絵を保存
            x_data = group["x"]
            y_data = group["y"]

        line, = ax.plot(x_data, y_data, label=f'ID:{id_num}')#描画した線を書き込む
        artists.append(line)#アニメーションに使うグラフをIDごとに保存
    frames.append(artists)#フレームを保存

ani = anime.ArtistAnimation(fig, frames, interval=10)#アニメーションを用意


plt.axis("equal")#アスペクト比を縦横同じに
plt.xlim(-100, 100)#交差点から100m範囲の動きを表示
plt.ylim(-100, 100)
plt.xlabel("East[m]")
plt.ylabel("North[m]")
im = Image.open("kousaten.png")#背景に交差点の画像を挿入
im = im.resize((770, 770))#サイズ調整
fig.figimage(im, -3, 24, zorder=1, alpha=0.5)#位置調整

plt.show()#アニメーション表示
