import os
import mne
import numpy as np
import time
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import Amplitude
import pandas as pd
import gtda
import re
from scipy import io
import time
import networkx as nx
import itertools

OutputDistributionFile = 'CycleRatio-main\\output'
sum_node = []

def eeg2feature(datapath, band):
    paths = datapath
    sfreq = 500
    channel = 29
    fstart = 0
    fend = 160
    result = []
    for path in paths:
        raw = mne.io.read_raw_edf(path,
                                     preload=True,
                                     )
        raww = raw.copy().resample(sfreq=500)
        channels_to_drop = raww.ch_names[-3:]
        raww.drop_channels(channels_to_drop)
        raww.filter(band[0], band[1], fir_design='firwin')  # 特定频段
        data = []

        t_idx1 = raww.time_as_index([0., 40.])
        data1, times1 = raww[:, t_idx1[0]:t_idx1[1]]
        data1 = data1.transpose()
        data1 = data1.reshape((1, 20000, 29))

        t_idx2 = raww.time_as_index([40., 80.])
        data2, times2 = raww[:, t_idx2[0]:t_idx2[1]]
        data2 = data2.transpose()
        data2 = data2.reshape((1, 20000, 29))

        t_idx3 = raww.time_as_index([80., 120.])
        data3, times3 = raww[:, t_idx3[0]:t_idx3[1]]
        data3 = data3.transpose()
        data3 = data3.reshape((1, 20000, 29))

        t_idx4 = raww.time_as_index([120., 160.])
        data4, times4 = raww[:, t_idx4[0]:t_idx4[1]]
        data4 = data4.transpose()
        data4 = data4.reshape((1, 20000, 29))

        data.append(data1)
        data.append(data2)
        data.append(data3)
        data.append(data4)

        # 每个人有四段数据
        Windows_index = 1
        for data_1 in data :
            from gtda.time_series import PearsonDissimilarity
            from gtda.homology import VietorisRipsPersistence
            from gtda.diagrams import Amplitude
            import time
            start = time.time()
            PD = PearsonDissimilarity()
            X_pd = PD.fit_transform(data_1)

            # 生成皮尔逊相关性图
            import seaborn
            # file_name = '5102病人'
            # file_path = 'I:\\GAT_graph_classification'
            # fig_path = f'{file_path}/{file_name}'
            X_pl = X_pd.reshape(29, 29)
            # 画图
            # fig = seaborn.heatmap(X_pl)
            # heatmap_fig = fig.get_figure()
            # 保存
            # heatmap_fig.savefig(fig_path, dpi = 400)

            # 得出组成复形的脑节点，并给出过滤阈值
            import gudhi
            import numpy as np
            correlation_matrix = X_pl
            distance_matrix = X_pl
            # 最大过滤尺度为1，最大维度为2
            rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=1.0)

            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
                         repr(simplex_tree.num_simplices()) + ' simplices - ' + \
                         repr(simplex_tree.num_vertices()) + ' vertices.'
            print(result_str)
            fmt = '%s -> %.2f'


            # 画出一个持续图，这里是复形
            diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
            # print("diag=", diag)
            # 画出一个持续图
            # gudhi.plot_persistence_diagram(diag)
            cha = []
            for shengsi in diag:
                chazhi = shengsi[1][1] - shengsi[1][0]
                cha.append(chazhi)
            cha.sort(reverse=True)

            cha = list(filter(lambda x: x <= 100000, cha))
            # print(cha)
            for shengsi in diag:
                if cha[0] == shengsi[1][1] - shengsi[1][0]:
                    top1 = shengsi
                    # print(top1)
                if cha[1] == shengsi[1][1] - shengsi[1][0]:
                    top2 = shengsi
                    # print(top2)
                if cha[2] == shengsi[1][1] - shengsi[1][0]:
                    top3 = shengsi
                    # print(top3)
                if cha[3] == shengsi[1][1] - shengsi[1][0]:
                    top4 = shengsi
                    # print(top4)
                if cha[4] == shengsi[1][1] - shengsi[1][0]:
                    top5 = shengsi
                    # print(top5)

            top1_node = []
            top2_node = []
            top3_node = []
            top4_node = []
            top5_node = []
            get_simplices = []
            for filtered_value in simplex_tree.get_simplices():
                get_simplices.append(filtered_value)
                if len(filtered_value[0]) == 3:
                    if (filtered_value[1] >= top1[1][0] and filtered_value[1] <= top1[1][1]):
                        top1_node.append(filtered_value[0])
                    if (filtered_value[1] >= top2[1][0] and filtered_value[1] <= top2[1][1]):
                        top2_node.append(filtered_value[0])
                    if (filtered_value[1] >= top3[1][0] and filtered_value[1] <= top3[1][1]):
                        top3_node.append(filtered_value[0])
                    if (filtered_value[1] >= top4[1][0] and filtered_value[1] <= top4[1][1]):
                        top4_node.append(filtered_value[0])
                    if (filtered_value[1] >= top5[1][0] and filtered_value[1] <= top5[1][1]):
                        top5_node.append(filtered_value[0])
            # print('这是所有的二维复形组成的脑节点>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # print(top1_node)
            # print(top2_node)
            # print(top3_node)
            # print(top4_node)
            # print(top5_node)
            # print(get_simplices)

            '''
            top1的重构图
            '''
            graph1 = []
            for two_dimensional_complex in top1_node:
                one = [two_dimensional_complex[0], two_dimensional_complex[1]]
                two = [two_dimensional_complex[0], two_dimensional_complex[2]]
                three = [two_dimensional_complex[1], two_dimensional_complex[2]]
                graph1.append(one)
                graph1.append(two)
                graph1.append(three)
            # 去重
            graph1 = list(set([tuple(t) for t in graph1]))
            # print(graph1)
            # 生成图
            Mygraph1 = nx.from_edgelist(graph1)
            # print(Mygraph1)
            # 去掉自环
            Mygraph1.remove_edges_from(nx.selfloop_edges(Mygraph1))

            '''
            top2的重构图
            '''
            graph2 = []
            for two_dimensional_complex in top2_node:
                one = [two_dimensional_complex[0], two_dimensional_complex[1]]
                two = [two_dimensional_complex[0], two_dimensional_complex[2]]
                three = [two_dimensional_complex[1], two_dimensional_complex[2]]
                graph2.append(one)
                graph2.append(two)
                graph2.append(three)
            # 去重
            graph2 = list(set([tuple(t) for t in graph2]))
            # print(graph2)
            # 生成图
            Mygraph2 = nx.from_edgelist(graph2)
            # print(Mygraph2)
            # 去掉自环
            Mygraph2.remove_edges_from(nx.selfloop_edges(Mygraph2))

            '''
            top3的重构图
            '''
            graph3 = []
            for two_dimensional_complex in top3_node:
                one = [two_dimensional_complex[0], two_dimensional_complex[1]]
                two = [two_dimensional_complex[0], two_dimensional_complex[2]]
                three = [two_dimensional_complex[1], two_dimensional_complex[2]]
                graph3.append(one)
                graph3.append(two)
                graph3.append(three)
            # 去重
            graph3 = list(set([tuple(t) for t in graph3]))
            # print(graph3)
            # 生成图
            Mygraph3 = nx.from_edgelist(graph3)
            # print(Mygraph3)
            # 去掉自环
            Mygraph3.remove_edges_from(nx.selfloop_edges(Mygraph3))

            '''
            top4的重构图
            '''
            graph4 = []
            for two_dimensional_complex in top4_node:
                one = [two_dimensional_complex[0], two_dimensional_complex[1]]
                two = [two_dimensional_complex[0], two_dimensional_complex[2]]
                three = [two_dimensional_complex[1], two_dimensional_complex[2]]
                graph4.append(one)
                graph4.append(two)
                graph4.append(three)
            # 去重
            graph4 = list(set([tuple(t) for t in graph4]))
            # print(graph4)
            # 生成图
            Mygraph4 = nx.from_edgelist(graph4)
            # print(Mygraph4)
            # 去掉自环
            Mygraph4.remove_edges_from(nx.selfloop_edges(Mygraph4))

            '''
            top5的重构图
            '''
            graph5 = []
            for two_dimensional_complex in top5_node:
                one = [two_dimensional_complex[0], two_dimensional_complex[1]]
                two = [two_dimensional_complex[0], two_dimensional_complex[2]]
                three = [two_dimensional_complex[1], two_dimensional_complex[2]]
                graph5.append(one)
                graph5.append(two)
                graph5.append(three)
            # 去重
            graph5 = list(set([tuple(t) for t in graph5]))
            # print(graph5)
            # 生成图
            Mygraph5 = nx.from_edgelist(graph5)
            # print(Mygraph5)
            # 去掉自环
            Mygraph5.remove_edges_from(nx.selfloop_edges(Mygraph5))

            Mygraph_list = []
            Mygraph_list.append(Mygraph1)
            Mygraph_list.append(Mygraph2)
            Mygraph_list.append(Mygraph3)
            Mygraph_list.append(Mygraph4)
            Mygraph_list.append(Mygraph5)


            '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            '''以上得到了graph1--graph5的图。是一个人一个时间段的五个不同的同调群的graph'''

            # Mygraph = Mygraph1
            # 这里是遍历每段数据的前top5的同调群
            for Mygraph in Mygraph_list:

                '''以下代码是求圈比'''
                NodeNum = Mygraph.number_of_nodes()
                print('Number of nodes = ', NodeNum)
                print("Number of deges:", Mygraph.number_of_edges())

                DEF_IMPOSSLEN = NodeNum + 1  # Impossible simple cycle length

                SmallestCycles = set()
                NodeGirth = dict()
                NumSmallCycles = 0
                CycLenDict = dict()
                CycleRatio = {}

                SmallestCyclesOfNodes = {}  #

                Coreness = nx.core_number(Mygraph)

                removeNodes = set()
                for i in Mygraph.nodes():  #
                    SmallestCyclesOfNodes[i] = set()
                    CycleRatio[i] = 0
                    if Mygraph.degree(i) <= 1 or Coreness[i] <= 1:
                        NodeGirth[i] = 0
                        removeNodes.add(i)
                    else:
                        NodeGirth[i] = DEF_IMPOSSLEN

                Mygraph.remove_nodes_from(removeNodes)  #
                NumNode = Mygraph.number_of_nodes()  # update

                for i in range(3, Mygraph.number_of_nodes() + 2):
                    CycLenDict[i] = 0

                def my_all_shortest_paths(G, source, target):
                    pred = nx.predecessor(G, source)
                    if target not in pred:
                        raise nx.NetworkXNoPath(
                            f"Target {target} cannot be reached" f"from given sources"
                        )
                    sources = {source}
                    seen = {target}
                    stack = [[target, 0]]
                    top = 0
                    while top >= 0:
                        node, i = stack[top]
                        if node in sources:
                            yield [p for p, n in reversed(stack[: top + 1])]
                        if len(pred[node]) > i:
                            stack[top][1] = i + 1
                            next = pred[node][i]
                            if next in seen:
                                continue
                            else:
                                seen.add(next)
                            top += 1
                            if top == len(stack):
                                stack.append([next, 0])
                            else:
                                stack[top][:] = [next, 0]
                        else:
                            seen.discard(node)
                            top -= 1

                def getandJudgeSimpleCircle(objectList):  #
                    numEdge = 0
                    for eleArr in list(itertools.combinations(objectList, 2)):
                        if Mygraph.has_edge(eleArr[0], eleArr[1]):
                            numEdge += 1
                    if numEdge != len(objectList):
                        return False
                    else:
                        return True

                def getSmallestCycles():
                    NodeList = list(Mygraph.nodes())
                    NodeList.sort()
                    # setp 1
                    curCyc = list()
                    for ix in NodeList[:-2]:  # v1
                        if NodeGirth[ix] == 0:
                            continue
                        curCyc.append(ix)
                        for jx in NodeList[NodeList.index(ix) + 1: -1]:  # v2
                            if NodeGirth[jx] == 0:
                                continue
                            curCyc.append(jx)
                            if Mygraph.has_edge(ix, jx):
                                for kx in NodeList[NodeList.index(jx) + 1:]:  # v3
                                    if NodeGirth[kx] == 0:
                                        continue
                                    if Mygraph.has_edge(kx, ix):
                                        curCyc.append(kx)
                                        if Mygraph.has_edge(kx, jx):
                                            SmallestCycles.add(tuple(curCyc))
                                            for i in curCyc:
                                                NodeGirth[i] = 3
                                        curCyc.pop()
                            curCyc.pop()
                        curCyc.pop()
                    # setp 2
                    ResiNodeList = []  # Residual Node List
                    for nod in NodeList:
                        if NodeGirth[nod] == DEF_IMPOSSLEN:
                            ResiNodeList.append(nod)
                    if len(ResiNodeList) == 0:
                        return
                    else:
                        visitedNodes = dict.fromkeys(ResiNodeList, set())
                        for nod in ResiNodeList:
                            if Coreness[nod] == 2 and NodeGirth[nod] < DEF_IMPOSSLEN:
                                continue
                            for nei in list(Mygraph.neighbors(nod)):
                                if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                                    continue
                                if not nei in visitedNodes.keys() or not nod in visitedNodes[nei]:
                                    visitedNodes[nod].add(nei)
                                    if nei not in visitedNodes.keys():
                                        visitedNodes[nei] = set([nod])
                                    else:
                                        visitedNodes[nei].add(nod)
                                    if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                                        continue
                                    Mygraph.remove_edge(nod, nei)
                                    if nx.has_path(Mygraph, nod, nei):
                                        for path in my_all_shortest_paths(Mygraph, nod, nei):
                                            lenPath = len(path)
                                            path.sort()
                                            SmallestCycles.add(tuple(path))
                                            for i in path:
                                                if NodeGirth[i] > lenPath:
                                                    NodeGirth[i] = lenPath
                                    Mygraph.add_edge(nod, nei)

                def StatisticsAndCalculateIndicators():  #
                    global NumSmallCycles
                    NumSmallCycles = len(SmallestCycles)
                    for cyc in SmallestCycles:
                        lenCyc = len(cyc)
                        CycLenDict[lenCyc] += 1
                        for nod in cyc:
                            SmallestCyclesOfNodes[nod].add(cyc)
                    for objNode, SmaCycs in SmallestCyclesOfNodes.items():
                        if len(SmaCycs) == 0:
                            continue
                        cycleNeighbors = set()
                        NeiOccurTimes = {}
                        for cyc in SmaCycs:
                            for n in cyc:
                                if n in NeiOccurTimes.keys():
                                    NeiOccurTimes[n] += 1
                                else:
                                    NeiOccurTimes[n] = 1
                            cycleNeighbors = cycleNeighbors.union(cyc)
                        cycleNeighbors.remove(objNode)
                        del NeiOccurTimes[objNode]
                        sum = 0
                        for nei in cycleNeighbors:
                            sum += float(NeiOccurTimes[nei]) / len(SmallestCyclesOfNodes[nei])
                        CycleRatio[objNode] = sum + 1

                def printAndOutput_ResultAndDistribution(objectList, nameString, Outpath):
                    addrespath = Outpath + nameString + '.txt'
                    Distribution = {}

                    for value in objectList.values():
                        if value in Distribution.keys():
                            Distribution[value] += 1
                        else:
                            Distribution[value] = 1

                    for (myk, myv) in Distribution.items():
                        Distribution[myk] = myv / float(NodeNum)
                    # list类型的数据，各个结点的圈比
                    rankedDict_ObjectList = sorted(objectList.items(), key=lambda d: d[1], reverse=True)
                    # 取圈比排名前10的节点
                    rankedDict_ObjectList_top10 = rankedDict_ObjectList[:10]
                    sum_node.append(rankedDict_ObjectList_top10)




                    # # 把节点的圈比写入文档
                    # fileout3 = open(addrespath, 'w')
                    # for d in range(len(rankedDict_ObjectList)):
                    #     fileout3.writelines("%6d %12.6f  \n" % (rankedDict_ObjectList[d][0], rankedDict_ObjectList[d][1]))
                    # fileout3.close()


                    # addrespath2 = Outpath + 'Distribution_' + nameString + '.txt'
                    # fileout2 = open(addrespath2, 'w')
                    # for (myk, myv) in Distribution.items():
                    #     fileout2.writelines("%12.6f %12.6f  \n" % (myk, myv))
                    # fileout2.close()

                # main fun
                StartTime = time.time()
                getSmallestCycles()
                EndTime1 = time.time()

                StatisticsAndCalculateIndicators()

                # output
                printAndOutput_ResultAndDistribution(CycleRatio, 'CycleRatio', OutputDistributionFile)

                # print('\nThe total number of the shortest basic cycles: %20d' % NumSmallCycles)
                print('Time: ', EndTime1 - StartTime)




            # """
            # 挑选出每个人最好的的时间段，（一个人共4段）
            # """
            # # 变为二维列表
            # rankedDict_ObjectList_top10 = list(itertools.chain.from_iterable(rankedDict_ObjectList_top10))
            #
            # # 只取节点编号
            # rankedDict_ObjectList_top10_bianhao = []
            # for node in rankedDict_ObjectList_top10:
            #     bianhao = node[0]
            #     rankedDict_ObjectList_top10_bianhao.append(bianhao)
            # print(rankedDict_ObjectList_top10_bianhao)
            # node_1 = 13
            # node_2 = 9
            # node_3 = 43
            # node_4 = 14
            # node_5 = 44
            # node_6 = 19
            # node_7 = 48
            # node_8 = 20
            # node_9 = 17
            # node_10 = 10
            # jioaji_sum = 0
            # if node_1 in rankedDict_ObjectList_top10_bianhao:
            #     jaioji_sum = jaioji_sum + 1


# 演示路径
# n_path = 'D:\\code\\exercise\\EEG_test\\teacher_test\\data\\data_23_split\\train16+16\\Theta(4-7Hz)\\'
# 实验路径
n_path = 'data/severe/'
paths = os.listdir(n_path)
n_path = [n_path + x for x in paths if x.endswith('.fif')]
delta_band = [1,3]
theta_band = [4,7]
alpha_band = [8,13]
beta_band = [14,30]
gamma_band = [31,40]
eeg2feature(n_path, delta_band)

# 变为二维列表
sum_node = list(itertools.chain.from_iterable(sum_node))
print(sum_node)

# 只取节点编号
all_bianhao =[]
for node in sum_node:
    bianhao = node[0]
    all_bianhao.append(bianhao)
print(all_bianhao)

# 对节点编号出现的次数进行统计
dict = {}
for key in all_bianhao:
    dict[key] = dict.get(key, 0) + 1
print(dict)

# 根据字典中脑节点出现的次数对脑节点进行排序
node_order = sorted(dict.items(),key=lambda x:x[1],reverse=True)
print(node_order)
# 把脑节点编号和脑节点圈比排名前十出现的次数写入一个文件中
fileout = open('cycle_ratio_patients_Beta_severe.txt', 'w')
for d in range(len(node_order)):
    fileout.writelines("%6d %12.6f\n" % (node_order[d][0], node_order[d][1]))
fileout.close()


print(sum(dict.values()))