-- REQ_01: 归集额预测输入取数逻辑
-- 目标：提取近 5 年（60个月）的历史数据及宏观/政策特征
SELECT 
    dt AS YCRQ,              -- 日期 (索引)
    YDGJJE,                 -- 月度归集额 (目标)
    GJJETBZZL,              -- 归集额同比增长率
    GJJEHBZZL,              -- 归集额环比增长率
    GJZHSL,                 -- 归集账户数量
    NF, YF, JD, SFJM, SFNM, -- 时间特征
    GDPZZL, JMSRSP,         -- 宏观经济指标
    GJZCDJZSJ, JCBLBH,      -- 政策特征
    CSRKQLQ, XZJY           -- 人口/就业数据
FROM 
    dw_hpf.ads_collection_forecasting_input
WHERE 
    dt >= add_months(current_date, -60) -- 取近5年数据
ORDER BY 
    dt ASC;
