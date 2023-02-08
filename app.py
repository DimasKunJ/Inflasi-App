import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

DATAPATH = "data/data_inflasi.csv"

st.set_page_config(layout="wide")


@st.cache
def load_data(PATH=DATAPATH):
    """_summary_

    Args:
        PATH (_type_, optional): _description_. Defaults to DATAPATH.

    Returns:
        _type_: _description_
    """
    data = pd.read_csv(PATH, parse_dates=[0], dayfirst=True)[::-1]
    data = data.set_index("Periode")
    return data


data = load_data()
data.index.freq = "MS"


def home():
    """_summary_
    Home Page
    """
    st.title("Tingkat Inflasi di Indonesia")
    date_format = "MMM-YYYY"  # format output
    start_date = data.index[0].to_pydatetime()
    end_date = data.index[-1].to_pydatetime()

    slider = st.slider(
        "Select date",
        min_value=start_date,
        value=(start_date, end_date),
        max_value=end_date,
        format=date_format,
    )

    part = data[slider[0] : slider[1]]
    col1, col2 = st.columns([1, 3])

    col2.line_chart(part, height=400)

    part.index = part.index.strftime("%Y-%m-%d")
    col1.write(part[::-1])

    st.sidebar.markdown(
        """> **Inflasi** adalah kecenderungan naiknya harga barang dan jasa secara umum yang \
             berlangsung secara terus-menerus. Tingkat inflasi digunakan sebagai indikator pertumbuhan dan stabilitas ekonomi."""
    )


def arima():
    """
    Create Arima
    """

    st.title("Peramalan Tingkat Inflasi di Indonesia dengan ARIMA")
    st.sidebar.markdown(
        """ Model ARIMA (p,d,q) merupakan model umum dari regresi deret waktu. \
            Pada model musiman, ARIMA memiliki parameter tambahan (P,D,Q,s).
        """
    )

    tab1, tab2, tab3 = st.columns([3, 0.5, 0.5])

    p = tab3.slider("Parameter p", min_value=0, max_value=5)
    d = tab3.slider("Parameter d", min_value=0, max_value=2)
    q = tab3.slider("Parameter q", min_value=0, max_value=5)

    col1, col2, col3, col4 = st.columns(4)
    P = tab2.slider("Parameter P", min_value=0, max_value=5)
    D = tab2.slider("Parameter D", min_value=0, max_value=2)
    Q = tab2.slider("Parameter Q", min_value=0, max_value=5)
    s = tab2.slider("Seasonal Order", min_value=0, max_value=24)

    model = ARIMA(data, order=(p, d, q), freq="MS", seasonal_order=(P, D, Q, s)).fit()
    y_pred = model.forecast(24)
    tab1.line_chart(y_pred)

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(float_format="%.2f").encode("utf-8")

    csv = convert_df(y_pred)

    tab3.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="Hasil Peramalan.csv",
        mime="text/csv",
    )


page_names_to_funcs = {
    "Home": home,
    "ARIMA": arima,
}

demo_name = st.sidebar.selectbox("Choose Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
