#!/bin/bash

# Set the cookie value as an environment variable
export CF_BM_COOKIE="3vZ_y6jqZDbGMKvJn2fSFGy599ARVaVuhmZBhHA02SY-1716646964-1.0.1.1-JowtmY277PntAxGllkWB3E3719ImYmNd.4cuXN7m3G19KvZ9YpjLIqPoFH4x8g5S793AlVgNdQmTlSmEGBG7EQ"

# Make the API request
response=$(curl -s -o /dev/null -w "%{http_code}" 'https://api.coingecko.com/api/v3/coins/maga-hat/market_chart?vs_currency=eur&days=2' \
  -H "accept: application/json" \
  -H "cookie: __cf_bm=$CF_BM_COOKIE" \
  -H "if-none-match: W/"073eb5331d024709e75290659e0874cf"" \
  -H "user-agent: MyCustomUserAgent")

# Check if the request was successful
if [ "$response" == "200" ]; then
  # Print the response body
  curl 'https://api.coingecko.com/api/v3/coins/maga-hat/market_chart?vs_currency=eur&days=2' \
    -H 'accept: application/json' \
    -H "cookie: __cf_bm=$CF_BM_COOKIE" \
    -H "if-none-match: W/"073eb5331d024709e75290659e0874cf"" \
    -H 'user-agent: MyCustomUserAgent'
else
  # Print an error message
  echo "Error: API request failed with status code $response"
fi
