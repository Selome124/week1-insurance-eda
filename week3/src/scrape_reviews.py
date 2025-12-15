from google_play_scraper import Sort, reviews_all
import pandas as pd

def scrape_reviews(app_id, app_name):
    print(f"Scraping reviews for {app_name}...")

    data = reviews_all(
        app_id,
        sleep_milliseconds=0,
        lang='en',
        country='us',
        sort=Sort.NEWEST
    )

    df = pd.DataFrame(data)
    df['app_name'] = app_name

    return df


if __name__ == "__main__":
    apps = {
        "com.combanketh.mobilebanking": "Commercial Bank of Ethiopia",
        "com.bankofabyssinia.boaapp": "Bank of Abyssinia",
        "com.awashbank.mobile": "Awash Bank"
    }

    all_reviews = pd.DataFrame()

    for app_id, app_name in apps.items():
        df = scrape_reviews(app_id, app_name)
        all_reviews = pd.concat([all_reviews, df], ignore_index=True)

    all_reviews.to_csv("reviews.csv", index=False)
    print("Saved to reviews.csv (Done!)")
