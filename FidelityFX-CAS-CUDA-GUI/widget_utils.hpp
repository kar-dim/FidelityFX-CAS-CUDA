#ifndef WIDGET_UTILS_HPP
#define WIDGET_UTILS_HPP

#include <concepts>
#include <QPixmap>
#include <QSize>
#include <QWidget>
#include <type_traits>

//Utility class for common widget operations
class WidgetUtils final
{
public:
    WidgetUtils() = delete;
    WidgetUtils(const WidgetUtils&) = delete;
    WidgetUtils& operator=(const WidgetUtils&) = delete;
    WidgetUtils(WidgetUtils&&) = delete;
    WidgetUtils& operator=(WidgetUtils&&) = delete;

	//Scale the pixmap to the target size
    static void scalePixmap(QPixmap &pixmap, const QSize targetImageSize);

	//Generic method to set visibility of multiple widgets
    template <typename... Widgets>
    requires (std::derived_from<std::remove_pointer_t<Widgets>, QWidget> && ...)
    static void setVisibility(bool value, Widgets... widgets)
    {
        (widgets->setVisible(value), ...);
    }
};

#endif // WIDGET_UTILS_HPP
