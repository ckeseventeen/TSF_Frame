-- REQ_01: 归集额**推理(月度跑批)**取数逻辑 — 短窗口版本
-- 对应训练 SQL: req_01_collection_amount.sql (拉近 5 年训练样本)
--
-- 设计原则:
--   * 训练用全量历史(60 个月)样本多, 训出来的模型更鲁棒
--   * 推理(月度跑批)只需要"够算 lag/rolling 特征 + 喂模型 base buffer"的近期数据
--     避免每月 Hive 全量扫描浪费 I/O
--
-- 窗口长度推导(必须 ≥ min_required_rows):
--   seq_len 1 + max(lag=12, rolling=12, diff=12) - 1 + pred_len 60 + 余量
--   = 1 + 12 - 1 + 60 + 余量
--   实际取 24 个月足够: 12 个月覆盖 lag/rolling, 余下 12 个月给 autoregressive
--   滚动外推时的协变量延续 / 监控比对参考
SELECT
    dt AS YCRQ,              -- 日期 (索引)
    YDGJJE,                  -- 月度归集额 (目标)
    GJJETBZZL,               -- 归集额同比增长率
    GJJEHBZZL,               -- 归集额环比增长率
    GJZHSL,                  -- 归集账户数量
    NF, YF, JD, SFJM, SFNM,  -- 时间特征
    GDPZZL, JMSRSP,          -- 宏观经济指标
    GJZCDJZSJ, JCBLBH,       -- 政策特征
    CSRKQLQ, XZJY            -- 人口/就业数据
FROM
    dw_hpf.ads_collection_forecasting_input
WHERE
    dt >= add_months(current_date, -24)   -- 推理只拉近 2 年, 训练 SQL 拉 -60
ORDER BY
    dt ASC;
