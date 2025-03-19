class SimKGCL(KnowledgeRecommender):
    """
    模型 SimKGCL
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimKGCL, self).__init__(config, dataset)

        # 加载数据集信息
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._user = dataset.inter_feat[dataset.uid_field]  # 训练用户数组
        self._item = dataset.inter_feat[dataset.iid_field]  # 训练物品数组

        # 加载参数信息
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.require_pow = config['require_pow']
        self.layer_cl = config['layer_cl']
        self.cl_rate = config['cl_rate']
        self.tau = config['tau']
        # self.tau = 0.2
        self.kg_drop_rate = config['kg_drop_rate']
        self.ig_drop_rate = config['ig_drop_rate']
        self.mess_drop_rate = config['mess_drop_rate']

        # 定义层和损失函数
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.latent_dim)
        self.relation_embedding = torch.nn.Embedding(self.n_relations + 1, self.latent_dim)

        self.message_drop = torch.nn.Dropout(self.mess_drop_rate)
        self.node_drop = SparseDropout(self.ig_drop_rate)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # 存储变量用于全排序评估加速
        self.restore_user_e = None
        self.restore_item_e = None

        # 生成中间数据
        self.norm_adj_matrix, self.user_item_matrix = self.get_norm_adj_mat()
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        self.all_hs = torch.LongTensor(self.kg_graph.row).to(self.device)
        self.all_ts = torch.LongTensor(self.kg_graph.col).to(self.device)
        self.all_rs = torch.LongTensor(self.kg_graph.data).to(self.device)

        # 参数初始化
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        """
        获取用户和物品的归一化交互矩阵
        """
        # 构建邻接矩阵
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # 归一化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        # 添加epsilon以避免除以零警告
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # 将归一化邻接矩阵转换为张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        L_ = L.tocsr()[: self.n_users, self.n_users:].tocoo()
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        SparseL_ = torch.sparse.FloatTensor(i_, data_, torch.Size(L_.shape))

        return SparseL.to(self.device), SparseL_.to(self.device)

    def get_ego_embeddings(self):
        """
        获取用户和物品/实体的嵌入向量并合并为嵌入矩阵
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def kg_agg(self, entity_emb, user_emb, relation_emb, all_h, all_t, all_r, inter_matrix, attention=True):
        """
        知识图谱聚合
        """
        from torch_scatter import scatter_softmax, scatter_mean

        n_entities = entity_emb.shape[0]
        edge_relation_emb = relation_emb[all_r]
        neigh_relation_emb = (entity_emb[all_t] * edge_relation_emb)

        if attention:
            neigh_relation_emb_weight = self.calculate_sim_hrt(
                entity_emb[all_h], entity_emb[all_t], edge_relation_emb
            )
            # [-1, 1] -> [-1, embedding_size]
            neigh_relation_emb_weight = neigh_relation_emb_weight.expand(
                neigh_relation_emb.shape[0], neigh_relation_emb.shape[1]
            )
            neigh_relation_emb_weight = scatter_softmax(
                neigh_relation_emb_weight, index=all_h, dim=0
            )  # [-1, embedding_size]
            neigh_relation_emb = torch.mul(
                neigh_relation_emb_weight, neigh_relation_emb
            )

        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=all_h, dim_size=n_entities, dim=0
        )
        user_agg = torch.sparse.mm(inter_matrix, entity_emb[:self.n_items])
        score = torch.mm(user_emb, relation_emb.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, relation_emb)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        """
        计算头实体、尾实体和关系嵌入向量之间的相似度
        """
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(
            head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)
        ).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

    def kg_forward(self, ego_embeddings, Drop=False):
        """
        知识图谱前向传播
        """
        user_emb, entity_emb = torch.split(ego_embeddings, [self.n_users, self.n_entities])
        # drop triplet edges in KG
        if Drop and self.kg_drop_rate > 0.0:
            all_h, all_t, all_r = self.edge_sampling(self.all_hs, self.all_ts, self.all_rs, 1 - self.kg_drop_rate)
        else:
            all_h, all_t, all_r = self.all_hs, self.all_ts, self.all_rs
        # drop interaction edges in IG
        if Drop and self.ig_drop_rate > 0.0:
            inter_matrix = self.node_drop(self.user_item_matrix)
        else:
            inter_matrix = self.user_item_matrix

        relation_emb = self.relation_embedding.weight

        entity_emb, user_emb = self.kg_agg(entity_emb, user_emb, relation_emb, all_h, all_t, all_r, inter_matrix)
        if Drop and self.mess_drop_rate > 0.0:
            entity_emb = self.message_drop(entity_emb)
            user_emb = self.message_drop(user_emb)

        entity_emb = F.normalize(entity_emb)
        user_emb = F.normalize(user_emb)

        return torch.cat([user_emb, entity_emb], dim=0)

    def forward(self, cl=False, Drop=False):
        """
        模型前向传播
        """
        ego_embeddings = self.get_ego_embeddings()

        all_ig_embeddings = []
        all_kg_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ig_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings[:self.n_users + self.n_items])
            kg_embeddings = self.kg_forward(all_kg_embeddings[-1], Drop=Drop)

            ego_embeddings = kg_embeddings
            ego_embeddings[:self.n_items + self.n_users] += ig_embeddings
            all_ig_embeddings.append(ego_embeddings)
            all_kg_embeddings.append(kg_embeddings)

        final_embeddings = torch.stack(all_ig_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        final_embeddings = final_embeddings[:self.n_users + self.n_items]
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])

        all_kg_embeddings = all_kg_embeddings
        final_kg_embeddings = torch.stack(all_kg_embeddings, dim=1)
        final_kg_embeddings = torch.mean(final_kg_embeddings, dim=1)
        final_kg_embeddings = final_kg_embeddings[:self.n_users + self.n_items]
        user_kg_embeddings, item_kg_embeddings = torch.split(final_kg_embeddings, [self.n_users, self.n_items])

        if cl:
            return user_all_embeddings, item_all_embeddings, user_kg_embeddings, item_kg_embeddings
        return user_all_embeddings, item_all_embeddings

    def edge_sampling(self, h_index, t_index, r_index, rate=0.5):
        """
        边缘采样
        """
        n_edges = h_index.shape[0]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return h_index[random_indices], t_index[random_indices], r_index[random_indices]

    def calculate_loss(self, interaction):
        """
        计算损失函数
        """
        # 清除存储变量
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, \
            user_kg_emb, item_kg_emb = self.forward(cl=True, Drop=True)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # 计算正则化损失
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.entity_embedding(pos_item)
        neg_ego_embeddings = self.entity_embedding(neg_item)
        reg_loss = (torch.norm(u_ego_embeddings, p=2) + torch.norm(pos_ego_embeddings, p=2) \
                    + torch.norm(neg_ego_embeddings, p=2)) * self.reg_weight

        # BPR损失
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)).sum()
        mf_loss = mf_loss + reg_loss

        ssl_loss2 = self.calculate_ssl_loss(user, pos_item, user_all_embeddings, user_kg_emb, item_all_embeddings,
                                            item_kg_emb)
        return mf_loss, ssl_loss2

    def calculate_ssl_loss(self, user, item, user_embeddings_v1,
                           user_embeddings_v2, item_embeddings_v1, item_embeddings_v2):
        """
        计算自监督损失
        """
        norm_user_v1 = F.normalize(user_embeddings_v1[torch.unique(user)])
        norm_user_v2 = F.normalize(user_embeddings_v2[torch.unique(user)])
        norm_item_v1 = F.normalize(item_embeddings_v1[torch.unique(item)])
        norm_item_v2 = F.normalize(item_embeddings_v2[torch.unique(item)])

        user_pos_score = torch.mul(norm_user_v1, norm_user_v2).sum(dim=1)
        user_ttl_score = torch.matmul(norm_user_v1, norm_user_v2.t())
        user_pos_score = torch.exp(user_pos_score / self.tau)
        user_ttl_score = torch.exp(user_ttl_score / self.tau).sum(dim=1)
        user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()

        item_pos_score = torch.mul(norm_item_v1, norm_item_v2).sum(dim=1)
        item_ttl_score = torch.matmul(norm_item_v1, norm_item_v2.t())
        item_pos_score = torch.exp(item_pos_score / self.tau)
        item_ttl_score = torch.exp(item_ttl_score / self.tau).sum(dim=1)
        item_ssl_loss = -torch.log(item_pos_score / item_ttl_score).sum()

        ssl_loss = user_ssl_loss + item_ssl_loss

        return ssl_loss * self.cl_rate

    def predict(self, interaction):
        """
        预测用户与物品的交互评分
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        完整排序预测
        """
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
