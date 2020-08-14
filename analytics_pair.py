import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

import download as dl


OUTPUT_NON_FILE = '../csv/no_match_pair_info.csv'
OUTPUT_MATCH_FILE = '../csv/match_pair_info.csv'


def search_pair(result_df, search_zero=True, n=100):
	'''
    マッチングが成功した数がnum以下のペアの情報を返す．

    Parameters
    ----------
    result_df : pandas.DataFrame
        走査した結果が入っているdf
    search_zero : boolean
        TRUEの場合マッチングが成立しなかったペア，FALSEの場合はマッチングが成立したペアの情報を返す．

    Returns
    -------
     : pandas.DataFrame
		 before_name, after_name, all_numの3列
	'''
	if search_zero:
		return result_df[result_df.all_num == 0]

	return result_df[result_df.all_num > 2000].sample(n=n)



def get_another_info(no_match_pair):
	'''
    NACを検索してNAC画像の情報を全て取得する．

    Parameters
    ----------
    no_match_pair : list of list of taple
		[[(before info: lat, lon, PRODUCT_ID), (after info)], ...]

    Returns
    -------
    no_match_df : pandas.DataFrame
		全ての情報が入っているデータフレーム．
	'''
	no_match_df = pd.DataFrame()
	for i, row in enumerate(no_match_pair.itertuples()):
		nacs_df = dl.get_data_from_point([row.lat, row.lon])
		before_info = nacs_df[nacs_df.PRODUCT_ID == row.before_name].iloc[:, 15:27]
		before_info.index = [i]
		before_info = before_info.add_prefix('BEFORE_')

		after_info = nacs_df[nacs_df.PRODUCT_ID == row.after_name].iloc[:, 15:27]
		after_info.index = [i]
		after_info = after_info.add_prefix('AFTER_')

		no_match_df_tmp = pd.concat([before_info, after_info], axis=1)
		no_match_df_tmp['DIF_RESOLUTION'] = abs(no_match_df_tmp['BEFORE_RESOLUTION'] - no_match_df_tmp['AFTER_RESOLUTION'])
		no_match_df_tmp['DIF_EMMISSION_ANGLE'] = abs(no_match_df_tmp['BEFORE_EMMISSION_ANGLE'] - no_match_df_tmp['AFTER_EMMISSION_ANGLE'])
		no_match_df_tmp['DIF_INCIDENCE_ANGLE'] = abs(no_match_df_tmp['BEFORE_INCIDENCE_ANGLE'] - no_match_df_tmp['AFTER_INCIDENCE_ANGLE'])
		no_match_df_tmp['DIF_PHASE_ANGLE'] = abs(no_match_df_tmp['BEFORE_PHASE_ANGLE'] - no_match_df_tmp['AFTER_PHASE_ANGLE'])
		no_match_df_tmp['DIF_NORTH_AZIMUTH'] = abs(no_match_df_tmp['BEFORE_NORTH_AZIMUTH'] - no_match_df_tmp['AFTER_NORTH_AZIMUTH'])
		if no_match_df.empty:
			no_match_df = no_match_df_tmp
		else:
			no_match_df = pd.concat([no_match_df, no_match_df_tmp])
	return no_match_df


def logistic(X_train, X_test, Y_train, Y_test):
	print('Start LogisticRegression')
	lr = LogisticRegression(solver='liblinear', penalty='l1', max_iter=200) # ロジスティック回帰モデルのインスタンスを作成
	# lr = LogisticRegression()
	lr = lr.fit(X_train, Y_train) # ロジスティック回帰モデルの重みを学習
	
	# weight出力
	print("coefficient = ", lr.coef_)
	print("intercept = ", lr.intercept_)

	Y_pred = lr.predict(X_test)
	Y_score = lr.predict_proba(X_test)[:, 1] # 検証データがクラス1に属する確率

	# weightが0でなかったものを表示
	print(np.array(X_train.columns)[np.where(lr.coef_[0]== 0)])
	show_result(Y_pred, Y_score, X_test, Y_test, 'logistic_mutch_result.png')


def tree(X_train, X_test, Y_train, Y_test):
	print('Start DecisionTreeClassifier')
	clf = DecisionTreeClassifier(max_depth = 3)
	clf = clf.fit(X_train, Y_train)

	Y_pred = clf.predict(X_test)
	Y_score = clf.predict_proba(X_test)[:, 1] # 検証データがクラス1に属する確率
	show_result(Y_pred, Y_score, X_test, Y_test, 'tree_mutch_result.png')
	f = export_graphviz(clf, out_file = 'tree_result.dot', feature_names = X_train.columns,
		class_names = ['no_match_pair', 'match_pair'], filled = True, rounded = True)


def random_forest(X_train, X_test, Y_train, Y_test):
	clf = RandomForestClassifier(max_depth=3, random_state=0)
	clf = clf.fit(X_train, Y_train)

	Y_pred = clf.predict(X_test)
	Y_score = clf.predict_proba(X_test)[:, 1] # 検証データがクラス1に属する確率
	show_result(Y_pred, Y_score, X_test, Y_test,  'random_fores_mutch_result.png')

	features = X_train.columns
	importances = clf.feature_importances_
	indices = np.argsort(importances)

	plt.figure(figsize=(20,10))
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), features[indices])
	plt.savefig('RandomForest.png')


def show_result(Y_pred, Y_score, X_test, Y_test, img_path):
	print('confusion matrix = \n', confusion_matrix(y_true=Y_test, y_pred=Y_pred))
	print('accuracy = ', accuracy_score(y_true=Y_test, y_pred=Y_pred))
	print('precision = ', precision_score(y_true=Y_test, y_pred=Y_pred))
	print('recall = ', recall_score(y_true=Y_test, y_pred=Y_pred))
	print('f1 score = ', f1_score(y_true=Y_test, y_pred=Y_pred))

	fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=Y_score)
	print('auc = ', roc_auc_score(y_true=Y_test, y_score=Y_score))

	plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc(fpr, tpr))
	plt.plot([0, 1], [0, 1], linestyle='--', label='random')
	plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label='ideal')
	plt.legend()
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.savefig(img_path)


def main(argv):
	# エラーチェック
	if len(argv) != 1:
		print('Input is some thing wrong. \n ex) python analytics_pair.py ./done_list_NC.csv')
		return 0

	# マッチングがうまくいかなかった時の情報を集める
	if os.path.isfile(OUTPUT_NON_FILE):
		# まとめた結果のcsv受付
		print('Reading {}...'.format(OUTPUT_NON_FILE))
		no_match_df = pd.read_csv(OUTPUT_NON_FILE, index_col=0)
	else:
		print('Making {}...'.format(OUTPUT_NON_FILE))
		# result.csvを読み込む
		result_df = pd.read_csv(argv[0], usecols=[1, 2, 4, 5, 9])
		# ペア0のlat, lon, PRODUCT_IDを得る
		no_match_pair = search_pair(result_df)
		# lat, lon, PRODUCT_IDから検索して他の情報を得る
		no_match_df = get_another_info(no_match_pair)
		# まとめた結果のcsv出力
		no_match_df.to_csv(OUTPUT_NON_FILE)
	print('Now, {} ready'.format(OUTPUT_NON_FILE))

	
	# マッチングがうまくいった時の情報を集める
	if os.path.isfile(OUTPUT_MATCH_FILE):
		# まとめた結果のcsv受付
		print('Reading {}...'.format(OUTPUT_MATCH_FILE))
		match_df = pd.read_csv(OUTPUT_MATCH_FILE, index_col=0)
	else:
		print('Making {}...'.format(OUTPUT_MATCH_FILE))
		# result.csvを読み込む
		result_df = pd.read_csv(argv[0], usecols=[1, 2, 4, 5, 9])
		# マッチングがうまくいっているのlat, lon, PRODUCT_IDを得る
		match_pair = search_pair(result_df, n=len(no_match_df))
		# lat, lon, PRODUCT_IDから検索して他の情報を得る
		match_df = get_another_info(match_pair)
		# まとめた結果のcsv出力
		match_df.to_csv(OUTPUT_MATCH_FILE)
	print('Now, {} ready'.format(OUTPUT_MATCH_FILE))


	
	X = pd.concat([no_match_df, match_df])
	Y = np.concatenate([np.zeros(len(no_match_df)), np.ones(len(match_df))])
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # 80%のデータを学習データに、20%を検証データにする

	logistic(X_train, X_test, Y_train, Y_test)
	print('********************************')
	tree(X_train, X_test, Y_train, Y_test)
	print('********************************')
	random_forest(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
	main(sys.argv[1:])