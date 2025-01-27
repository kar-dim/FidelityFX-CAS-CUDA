#ifndef WIDGET_UTILS_HPP
#define WIDGET_UTILS_HPP

#include <QPixmap>
#include <QWidget>
//#include <concepts> //c++20 is broken in Qt for now

class WidgetUtils final
{
public:
    WidgetUtils() = delete;
    WidgetUtils(const WidgetUtils&) = delete;
    WidgetUtils& operator=(const WidgetUtils&) = delete;
    WidgetUtils(WidgetUtils&&) = delete;
    WidgetUtils& operator=(WidgetUtils&&) = delete;

    static void scalePixmap(QPixmap &pixmap, const QSize targetImageSize);
    template <typename... Widgets>
    //requires (std::same_as<Widgets, QWidget> && ...)
    static void setVisibility(bool value, Widgets... widgets)
    {
        (widgets->setVisible(value), ...);
    }
};

#endif // WIDGET_UTILS_HPP
