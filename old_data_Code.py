
# def get_data():
#     if redownload:
#         # bbox = BboxSelector([ # It defines a bbox for the area of Münster (Germany)
#         #     7.601165771484375,  # min_x
#         #     51.94807412325402,  # min_y
#         #     7.648200988769531,  # max_x
#         #     51.97261482608728  # max_y
#         # ])
#         # track_df = track_api.get_tracks(bbox=bbox, num_results=50)
#         # print("shape::", track_df.shape)
#         # print("track_df.head()",track_df)
#         # track_df.to_csv('track_df2.csv', index=False)
#         endpoint = "https://envirocar.org/api/stable/tracks"
#         bbox = " 7.601165771484375,51.94807412325402,7.648200988769531,51.97261482608728"  # It defines a bbox for the area of Münster (Germany)
#
#         return track_df
#     else:
#         track_df = df = pd.read_csv('track_df.csv')
#         return track_df



# def get_time_sequences(track_df):
#     return
#     # if redownloadTS:
#     #     p_columns, _ = get_columns(track_df)
#     #     need_columns = False
#     #     all_columns = p_columns
#     #     ts = pd.DataFrame(columns=all_columns)
#     #     print("all_columns", all_columns)
#     #     unique_ids = track_df['track.id'].unique()
#     #     for track_id in unique_ids:
#     #         #print(index, row, row.dtype)
#     #         tr = track_df[track_df['track.id'] == track_id]
#     #         track = track_api.get_track(track_id)
#     #         print("tr", tr)
#     #         print(track)
#     #         print("track shape", track_id, track.shape)
#     #         fig = plt.figure()
#     #         tr.plot()
#     #         fig.show()
#     #         #print("tr['Speed.value']", tr['Speed.value'])
#     #
#     #         #print("tr['Speed.value']", tr['Speed.value'])
#     #         if need_columns:
#     #             t_columns, _ = get_columns(track)
#     #             all_columns.extend(t_columns)
#     #             print("track columsn", t_columns)
#     #             print("all_col", all_columns)
#     #             ts = pd.DataFrame(columns=t_columns)
#     #             need_columns = False
#     #         ts.append(track)
#     #         break
#     #     ts.to_csv('ts.csv', index=False)
#     #     print("shape", ts.shape)
#     #     print("all_col", get_columns(ts))
#     #     return ts
#     # else:
#     #     ts = pd.read_csv('ts.csv')
#     #     return ts



# final_df = pd.concat(dfs, ignore_index=True, sort=False)
#     final_df = final_df.loc[:, ~final_df.columns.duplicated()]
#     final_df = final_df.drop_duplicates()
#     final_df = final_df.reindex(sorted(final_df.columns), axis=1)


# def read_data(ids):
#     dfs = []
#     track_df = global_track_df_object
#     excluded_columns = ['id','time','track.begin','track.end']
#     sequences = []
#     for track_id in ids:
#         track_data = track_df[track_df['track.id'] == track_id]
#         track_data = track_data.drop(columns=excluded_columns, errors='ignore')
#         csv_file = f'envirocar_tracks_csv/{track_id}.csv'
#         df = pd.read_csv(csv_file,delimiter=';')
#         df.fillna(df.mean(numeric_only=True), inplace=True)
#         print("type", type(df), df.shape)
#         zeroth = track_data.iloc[0]
#         mean_v = track_data.median(numeric_only=True)
#         # print("zeroth", zeroth)
#         # print("mean_v",mean_v)
#         zeroth.fillna(mean_v, inplace=True)
#         track_sequences = []
#         for row in df.itertuples():
#             print("type row", row)
#             break
#             updated_row = pd.concat([row,zeroth])
#             print("updated", type(updated_row),get_columns(updated_row))
#             break
#             track_sequences.append(updated_row)
#         track_final_df = pd.concat(track_sequences)
#         print("shapess: track_final_df",type(track_final_df),track_final_df.shape)
#         sequences.append(track_final_df)
#     return sequences