from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from FileHandling.DownloadAndSave import DownloadAndSave

class DoScrape:

    def __init__(self) -> None:
        pass

    def fakecaptcha_interaction(self, numbers, iteration):
        ds = DownloadAndSave()
        url = 'https://fakecaptcha.com/'
        options = Options()
        options.headless = True
        browser = webdriver.Firefox(options=options)
        browser.get(url)
        browser.implicitly_wait(3)
        search_field = browser.find_element_by_css_selector("input[name='words'][type='text']")
        search_field.send_keys(numbers)
        search_field.submit()
        browser.implicitly_wait(6)
        WebDriverWait(browser, 1000000).until(EC.element_to_be_clickable((By.XPATH, '/html/body/section[2]/div/div[2]/a'))).click()
        img = search_field = browser.find_element_by_css_selector("img[id='words']")
        link = img.get_attribute('src')
        ds.makeDownload(link, "/home/jovyan/Python_eksamen/Images/Scraped/" + str(numbers)+"("+str(iteration)+").jpg")
        