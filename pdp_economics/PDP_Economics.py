import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    ### PDP Economics
    TODO
    ### Want to learn more?

    - Check out [CryptoEconLab](https://cryptoeconlab.io)

    - Engage with us on [X](https://x.com/cryptoeconlab)

    - Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

    ### Disclaimer
    CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)