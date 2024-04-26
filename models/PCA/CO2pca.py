class CO2pca:
    def doPCA_CO2(numcomponents=2):
        '''
        param: number of components to reduce to, default: 2
        returns: reduced components, labels
        '''
        fin = None
        scaled = None
        max_len = 0
        file_name = ''
        for a in constants.all_columns_present_id_list:
            # print(a)
            df = pd.read_csv("imputed_envirocar_data"+'/'+a+'.csv')
            dd_ = df[df.columns[1:-1]]
            corr_matrix = dd_.corr()
            indices = corr_matrix[abs(corr_matrix['CO2(kg/h)']) > 0.5]['CO2(kg/h)'].keys()[1:]
            if len(indices) < numcomponents:
                continue
            # print(len(indices))
            scaled = pd.DataFrame(StandardScaler().fit_transform(dd_[indices]))
            pca = PCA(n_components = numcomponents)
            pca.fit(scaled)
            df_pca = pca.transform(scaled)
            df_pca = pd.DataFrame(df_pca)
            df_pca['CO2(kg/h)'] = df['CO2(kg/h)'] #
            if len(df_pca) > max_len:
                max_len = len(df_pca)
                fin = df_pca
                file_name = a
        return fin[fin.columns[:-1]].to_numpy(), fin['CO2(kg/h)'].to_numpy()

X, y = doPCA_CO2(4)