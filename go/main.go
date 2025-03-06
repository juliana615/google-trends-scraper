package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/url"
	"strconv"

	http "github.com/bogdanfinn/fhttp"
	tls_client "github.com/bogdanfinn/tls-client"
	"github.com/bogdanfinn/tls-client/profiles"
)

type TokenResponseKeyword struct {
	Keyword string `json:"keyword"`
	Name    string `json:"name"`
	Type    string `json:"type"`
}
type Widget struct {
	Id      string `json:"id"`
	Title   string `json:"title"`
	Token   string `json:"token"`
	Request []byte `json:"request"`
}

// type TimeseriesWidget struct {

// }
type TokenResponse struct {
	Keywords     []TokenResponseKeyword `json:"keywords"`
	ShareText    string                 `json:"shareText"`
	ShowMultiple bool                   `json:"shouldShowMultiHeatMapMessage"`
	TimeRanges   []string               `json:"timeRanges"`
	Widgets      []byte                 `json:"widgets"`
}

type InputFilter struct {
	Keyword   string `json:"keyword"`
	Country   string `json:"country"`
	Timeframe string `json:"timeframe"`
	Category  int    `json:"category"`
	Gprop     string `json:"gprop"`
}

var BASE_TRENDS_URL = "https://trends.google.com/trends"
var EXPLORE_URL = fmt.Sprintf("%s/api/explore", BASE_TRENDS_URL)
var INTEREST_OVER_TIME_URL = fmt.Sprintf("%s/api/widgetdata/multiline", BASE_TRENDS_URL)
var INTEREST_BY_REGION_URL = fmt.Sprintf("%s/api/widgetdata/comparedgeo", BASE_TRENDS_URL)
var RELATED_QUERIES_URL = fmt.Sprintf("%s/api/widgetdata/relatedsearches", BASE_TRENDS_URL)

func getResponse(client tls_client.HttpClient, req *http.Request, header http.Header) (string, error) {
	req.Header = header

	resp, err := client.Do(req)
	if err != nil {
		log.Println(err)
		return "", err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("request failed. Status code: %d", resp.StatusCode)
	}

	readBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Println(err)
		return "", err
	}

	log.Println(string(readBytes))
	return string(readBytes), nil
}

func getData(client tls_client.HttpClient, httpMethod string, urlString string, header http.Header, trimCount int) (map[string]interface{}, error) {
	req, err := http.NewRequest(httpMethod, urlString, nil)
	if err != nil {
		log.Println(err)
		return nil, err
	}

	response, err := getResponse(client, req, header)
	if err != nil {
		log.Println(err)
		return nil, err
	}

	response_trimed := response[trimCount:]

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(response_trimed), &data); err != nil {
		fmt.Println("Error unmarshaling JSON:", err)
		return nil, err
	}

	return data, nil
}

func getToken(client tls_client.HttpClient, input InputFilter) (map[string]interface{}, error) {
	hl := "en-US"
	tz := "-480"
	req := fmt.Sprintf(
		`{"comparisonItem":[{"keyword":"%s","geo":"%s","time":"%s"}],"category":%s,"property":"%s"}`,
		input.Keyword, input.Country, input.Timeframe, strconv.Itoa(input.Category), input.Gprop,
	)
	payload := fmt.Sprintf(
		`?hl=%s&tz=%s&req=%s`,
		hl, tz, url.QueryEscape(req),
	)
	urlString := EXPLORE_URL + payload
	fmt.Println("Token urlString: ", urlString)
	referer := fmt.Sprintf("%s/explore?q=%s&hl=%s", BASE_TRENDS_URL, input.Keyword, hl[:2])
	fmt.Println("referer:", referer)
	header := http.Header{
		"accept":                      {"application/json, text/plain, */*"},
		"accept-language":             {"en-US,en;q=0.9"},
		"content-type":                {"application/json;charset=UTF-8"},
		"priority":                    {"u=1, i"},
		"sec-ch-ua":                   {"\"Not(A:Brand\";v=\"99\", \"Google Chrome\";v=\"133\", \"Chromium\";v=\"133\""},
		"sec-ch-ua-arch":              {"\"x86\""},
		"sec-ch-ua-bitness":           {"\"64\""},
		"sec-ch-ua-form-factors":      {"\"Desktop\""},
		"sec-ch-ua-full-version":      {"\"133.0.6943.142\""},
		"sec-ch-ua-full-version-list": {"\"Not(A:Brand\";v=\"99.0.0.0\", \"Google Chrome\";v=\"133.0.6943.142\", \"Chromium\";v=\"133.0.6943.142\""},
		"sec-ch-ua-mobile":            {"?0"},
		"sec-ch-ua-model":             {"\"\""},
		"sec-ch-ua-platform":          {"\"Windows\""},
		"sec-ch-ua-platform-version":  {"\"15.0.0\""},
		"sec-ch-ua-wow64":             {"?0"},
		"sec-fetch-dest":              {"empty"},
		"sec-fetch-mode":              {"cors"},
		"sec-fetch-site":              {"same-origin"},
		"x-client-data":               {"CLC1yQEIh7bJAQiitskBCKmdygEI7ODKAQiUocsBCImjywEIhaDNAQjpqc4BCL3VzgEIgNbOAQi64M4BCK7kzgEIjOXOARiPzs0BGOHizgE="},
		"cookie":                      {"__utmc=10102256; __utma=10102256.1127519021.1740957602.1741269669.1741275093.22; __utmz=10102256.1741275093.22.12.utmcsr=trends.google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; __utmb=10102256.1.10.1741275093; HSID=Axd4sT7jgiQJZxI35; SSID=ALG-6ly3f79ZskloZ; APISID=JOXLujcJUcSXPJ3h/ATDn9rFkuWpGmKwjn; SAPISID=0nutfgB2KiwE5Kgd/AhQ1IC-3dcxf77kpP; __Secure-1PAPISID=0nutfgB2KiwE5Kgd/AhQ1IC-3dcxf77kpP; __Secure-3PAPISID=0nutfgB2KiwE5Kgd/AhQ1IC-3dcxf77kpP; SEARCH_SAMESITE=CgQIrJ0B; SID=g.a000uAhwK_O9UJ94up-VgKYGSEzGKADY8lsN3Z4zExDYunjWOL9d2CrFKAjQb5WVHGV3EJeSeQACgYKAS8SARMSFQHGX2MijWsBAyGEV5FkAsSwNI4z1xoVAUF8yKpCbtylgAdCpGMkIGSL1aWp0076; __Secure-1PSID=g.a000uAhwK_O9UJ94up-VgKYGSEzGKADY8lsN3Z4zExDYunjWOL9dTivc0yVtbBesMgs0vQdIcAACgYKAWQSARMSFQHGX2MiUp2t6Yzi91KCrrTWoqtgARoVAUF8yKpWXB41sVSq_cqvvujj9JWS0076; __Secure-3PSID=g.a000uAhwK_O9UJ94up-VgKYGSEzGKADY8lsN3Z4zExDYunjWOL9dD8dQlEJXf6pLksCXjBOC4gACgYKAXMSARMSFQHGX2MiLnCPNQvAmi2boKzPKMk7cRoVAUF8yKpguI892pEF08fECnTOjA3J0076; _gid=GA1.3.252607514.1740957604; OTZ=7977560_24_24__24_; AEC=AVcja2d45h3p4qZHVoURGrGxT51ID7CibwMIHEphd_l57QfIvoLIEyAXKzg; NID=522=EGgzmgRXCVOCGqMExXZ5v6akZehNAKPXXahCMCU1aKuI9rIoXRP0_czVrDAeCpfGH3Qb2tLVb20PoGuER9grRKFhpDP8ZzWiwzg2-z6PIc_yLVXyI6ju7RiR_EfWcxWlR-lF0cXxceVPGV_egUhIyI4QHFYxHjTSTpS_Y2D96QfLgxk-eZ0CfRZpaa8wYU2_WbZCr2jqesf4McuaxRPXauy97J1buoXsq0GHoJZoSC8yoTfzRm2lZ7ElhkTeUsKC4X1LCWGOW_SA1X0FGeLFq_uGrum1kgGrtToZU-25POfbRyDO9XD0euXJ7DddkNrQBsry3FFTE4W-3ka8LzOQb88dn3lvw2Uj6ItWCRzpHwlhTjygg4ZEUawsYgo_0CEu19HKiCsymc2dR0zsZ45CaQAMYRlh6M56SH5W-a06PG4B0Wd52P_KLHikiWtOUu8TRCT15NkpCvam7zd--WvDT2BN_hGrOa1IpLsVt7VCvTNzoisViehFKn2xaHdOC5GELRZAFi83SXyCL4wzRO14RJE022-gvJ7bYUnrcp4kuhwytOMBdvIG3iteSkw_S0VTj6p1sYgWebOU1KJOOiekvcmpWyiLb0a_bC_wBGlUwGj-BH9N8nfg2aBAxz7A6wEuVbNEVd0PlSvjZaKlHMrU3SHhKhClS3H9ln2j_E4p0r_r7J6OizAujOG5LHSEYYRJdqcIDfmmQSXyB660xh-UD8zDq7kiXB2xL2N4Qxok5GJo-ewDH0RzoMJMz2QVktUb0Zvpsu_R47ONjivzoLQDqJ4cwLKSX5IcxEdCmZQPrkHYZMkISnr3Md9r3afy0AW55dykmxnyig1EmGs-uEzGfkjMBH0PksgwtccylNFjcOjDXQqojUU12bnfh1GN; __Secure-1PSIDTS=sidts-CjIBEJ3XVzQNQ_wROmn_ZGOBtZP9WO4jLA6yc7fA7E-Ae6y1FHUris42SXwV9FkthshnIhAA; __Secure-3PSIDTS=sidts-CjIBEJ3XVzQNQ_wROmn_ZGOBtZP9WO4jLA6yc7fA7E-Ae6y1FHUris42SXwV9FkthshnIhAA; _ga=GA1.1.1127519021.1740957602; _ga_VWZPXDNJJB=GS1.1.1741275090.26.1.1741275094.0.0.0; SIDCC=AKEyXzXbo9aZ6lN1I1K8nQfofkG9HtgTeMZSucJIPw1UEnj5DH8eLhCqCQ0rjucSETJJuNgQZQ0; __Secure-1PSIDCC=AKEyXzV4D9MngtHMJ9Zfw_Pdej7gK3C-YXhwJiYXK_V2CSSzKOQM_2TZb-rpJb1-_losarPlc5w; __Secure-3PSIDCC=AKEyXzVSUQ5aMnVHN7gfnOun4CriClXYzL4SP3yYv-EYkl2wtE0pBh7eHFcoehRbydWduAFkUEF0"},
		"Referer":                     {referer},
		"Referrer-Policy":             {"strict-origin-when-cross-origin"},
		"user-agent":                  {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"},
		http.HeaderOrderKey: {
			"accept",
			"accept-language",
			"content-type",
			"priority",
			"sec-ch-ua",
			"sec-ch-ua-arch",
			"sec-ch-ua-bitness",
			"sec-ch-ua-form-factors",
			"sec-ch-ua-full-version",
			"sec-ch-ua-full-version-list",
			"sec-ch-ua-mobile",
			"sec-ch-ua-model",
			"sec-ch-ua-platform",
			"sec-ch-ua-platform-version",
			"sec-ch-ua-wow64",
			"sec-fetch-dest",
			"sec-fetch-mode",
			"sec-fetch-site",
			"x-client-data",
			"cookie",
			"Referer",
			"Referrer-Policy",
			"user-agent",
		},
	}

	data, err := getData(client, http.MethodPost, urlString, header, 4)
	if err != nil {
		log.Println(err)
		return nil, err
	}
	return data, nil
}

func main() {
	jar := tls_client.NewCookieJar()
	options := []tls_client.HttpClientOption{
		tls_client.WithTimeoutSeconds(30),
		tls_client.WithClientProfile(profiles.Chrome_133),
		tls_client.WithNotFollowRedirects(),
		tls_client.WithCookieJar(jar), // create cookieJar instance and pass it as argument
	}

	client, err := tls_client.NewHttpClient(tls_client.NewNoopLogger(), options...)
	if err != nil {
		log.Println(err)
		return
	}

	input := InputFilter{
		Keyword:   "startup",
		Country:   "",
		Timeframe: "today 12-m",
		Category:  0,
		Gprop:     "",
	}

	token, err := getToken(client, input)
	if err != nil {
		log.Println(err)
		return
	}
	fmt.Println("token: ", token)
}
